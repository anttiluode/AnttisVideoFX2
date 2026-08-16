"""CUDA/torch PhaseRail engine used by the layered AI-video effect.

This is the reusable, GUI-free part of Antti's PhaseRail:

* a fixed 5-scale x 6-orientation complex Gabor pyramid
* coarse-to-fine phase unwrapping
* per-pixel binding of oriented motion into one 2-D displacement
* persistent transport of a generated appearance field
* optional affine-nullspace protection when a new generated keyframe arrives

The surrounding effect is responsible for segmentation and compositing.  This
module knows nothing about faces, masks, prompts, or backgrounds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    import torch
    TORCH_OK = True
except Exception:  # pragma: no cover - app still opens without torch
    torch = None
    TORCH_OK = False

EPS = 1e-7
TAU = 2.0 * math.pi


class ReferenceGaborBank:
    """Build the exact frequency masks once in numpy.

    Torch uses these masks for batched FFT inference.  Keeping construction in
    numpy makes the bank deterministic and keeps the implementation readable.
    """

    def __init__(
        self,
        size: int = 128,
        center_frequencies: tuple[float, ...] = (0.32, 0.18, 0.10, 0.055, 0.03),
        orientations: int = 6,
        sigma_on_frequency: float = 0.60,
    ) -> None:
        if size < 32:
            raise ValueError("rail size must be at least 32")
        if orientations < 3:
            raise ValueError("orientations must be at least 3")

        self.size = int(size)
        self.freqs = np.asarray(center_frequencies, np.float32)
        self.num_scales = len(self.freqs)
        self.num_orient = int(orientations)
        self.thetas = np.arange(self.num_orient, dtype=np.float32) * (
            np.pi / self.num_orient
        )

        fy = np.fft.fftfreq(self.size).astype(np.float32)
        fx = np.fft.fftfreq(self.size).astype(np.float32)
        FX, FY = np.meshgrid(fx, fy)
        radius = np.sqrt(FX * FX + FY * FY)
        angle = np.arctan2(FY, FX)

        low_raw = np.exp(-((radius / 0.020) ** 4)).astype(np.float32)
        theta_sigma = (np.pi / self.num_orient) / 1.35
        raw_bands: list[np.ndarray] = []
        meta: list[tuple[int, int, float]] = []

        for si, f0 in enumerate(self.freqs):
            log_ratio = np.log((radius + 1e-9) / float(f0))
            radial = np.exp(
                -(log_ratio * log_ratio)
                / (2.0 * math.log(sigma_on_frequency) ** 2)
            )
            radial[radius < 1e-8] = 0.0
            for oi, theta in enumerate(self.thetas):
                delta = ((angle - theta + np.pi / 2) % np.pi) - np.pi / 2
                angular = np.exp(-(delta * delta) / (2.0 * theta_sigma * theta_sigma))
                raw_bands.append((radial * angular).astype(np.float32))
                meta.append((si, oi, float(theta)))

        raw = np.stack(raw_bands, axis=0)
        covered = low_raw + raw.sum(axis=0)
        residual_raw = np.clip(1.0 - covered, 0.0, 1.0)
        total = covered + residual_raw + EPS

        self.low_filter = (low_raw / total).astype(np.float32)
        self.residual_filter = (residual_raw / total).astype(np.float32)
        self.even_filters = (raw / total[None]).astype(np.float32)
        self.meta = meta

        odd_filters: list[np.ndarray] = []
        kx: list[float] = []
        ky: list[float] = []
        for band, (si, _, theta) in zip(self.even_filters, meta):
            projection = FX * math.cos(theta) + FY * math.sin(theta)
            odd_filters.append((-1j * np.sign(projection) * band).astype(np.complex64))
            omega = TAU * float(self.freqs[si])
            kx.append(omega * math.cos(theta))
            ky.append(omega * math.sin(theta))

        self.odd_filters = np.stack(odd_filters, axis=0)
        self.kx = np.asarray(kx, np.float32)
        self.ky = np.asarray(ky, np.float32)

    @property
    def num_bands(self) -> int:
        return self.num_scales * self.num_orient


class TorchGaborPyramid:
    def __init__(self, size: int = 128, orientations: int = 6, device: str = "cuda") -> None:
        if not TORCH_OK:
            raise RuntimeError("PhaseRail requires torch")
        requested = device
        if requested == "cuda" and not torch.cuda.is_available():
            requested = "cpu"
        self.device = torch.device(requested)

        ref = ReferenceGaborBank(size=size, orientations=orientations)
        self.size = ref.size
        self.num_scales = ref.num_scales
        self.num_orient = ref.num_orient

        def t(a: np.ndarray, dtype):
            return torch.from_numpy(np.ascontiguousarray(a)).to(self.device, dtype=dtype)

        def hermitian(F: np.ndarray) -> np.ndarray:
            flipped = np.roll(np.flip(F, axis=(-2, -1)), 1, axis=(-2, -1))
            return 0.5 * (F + np.conj(flipped))

        analytic = hermitian(ref.even_filters.astype(np.complex64)) + 1j * hermitian(
            ref.odd_filters
        )
        self.analytic = t(analytic, torch.complex64)
        self.low_filter = t(ref.low_filter, torch.float32)
        self.residual_filter = t(ref.residual_filter, torch.float32)

        bank = torch.zeros(
            (2 + ref.num_bands, self.size, self.size),
            dtype=torch.complex64,
            device=self.device,
        )
        bank[0] = self.low_filter.to(torch.complex64)
        bank[1] = self.residual_filter.to(torch.complex64)
        bank[2:] = self.analytic
        self.bank = bank

        self.freqs = t(ref.freqs, torch.float32)
        self.thetas = t(ref.thetas, torch.float32)
        self.kx = t(ref.kx, torch.float32)
        self.ky = t(ref.ky, torch.float32)

        coords = (
            torch.arange(self.size, device=self.device, dtype=torch.float32)
            - 0.5 * (self.size - 1)
        ) / max(self.size - 1, 1)
        self.Y, self.X = torch.meshgrid(coords, coords, indexing="ij")

    @property
    def num_bands(self) -> int:
        return self.num_scales * self.num_orient

    def to_device(self, array: np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(array)).to(
            self.device, dtype=torch.float32
        )

    def analyze_stack(self, planes):
        spectrum = torch.fft.fft2(planes.to(torch.complex64))
        filtered = torch.fft.ifft2(spectrum[:, None] * self.bank[None])
        return filtered[:, 2:], filtered[:, 0].real, filtered[:, 1].real

    def analyze_gray(self, image):
        z, low, residual = self.analyze_stack(image[None])
        return z[0], low[0], residual[0]

    def analyze_bgr(self, bgr):
        planes = bgr.permute(2, 0, 1) if bgr.shape[-1] == 3 else bgr
        return self.analyze_stack(planes.contiguous())

    def analyze_bgr_low_residual(self, bgr):
        planes = bgr.permute(2, 0, 1) if bgr.shape[-1] == 3 else bgr
        spectrum = torch.fft.fft2(planes.contiguous().to(torch.complex64))
        low = torch.fft.ifft2(spectrum * self.low_filter[None]).real
        residual = torch.fft.ifft2(spectrum * self.residual_filter[None]).real
        return low, residual

    def synthesize_bgr(self, z, low, residual):
        out = low + residual + z.real.sum(dim=1)
        return out.permute(1, 2, 0).clamp(0.0, 1.0)

    def synthesize_gray(self, z, low, residual):
        return low + residual + z.real.sum(dim=0)

    def cross_scale_coherence(self, z):
        shaped = z.reshape(
            self.num_scales, self.num_orient, self.size, self.size
        )
        return (shaped.sum(dim=0).abs() / (shaped.abs().sum(dim=0) + EPS)).clamp(
            0.0, 1.0
        )

    def estimate_bound_phase_flow(self, z_now, z_previous, max_displacement: float = 4.0):
        S, O, H = self.num_scales, self.num_orient, self.size
        now = z_now.reshape(S, O, H, H)
        prev = z_previous.reshape(S, O, H, H)

        raw_phase = torch.angle(now * torch.conj(prev))
        amplitude = torch.sqrt(now.abs() * prev.abs())
        robust = torch.quantile(amplitude.flatten().float(), 0.90) + EPS
        amplitude_weight = (amplitude / robust).clamp(0.0, 1.0)

        displacement = raw_phase[-1] / (TAU * self.freqs[-1])
        accumulated = amplitude_weight[-1] + 1e-3
        for si in range(S - 2, -1, -1):
            expected = TAU * self.freqs[si] * displacement
            unwrapped = raw_phase[si] + TAU * torch.round((expected - raw_phase[si]) / TAU)
            candidate = unwrapped / (TAU * self.freqs[si])
            weight = amplitude_weight[si]
            displacement = (accumulated * displacement + weight * candidate) / (
                accumulated + weight + EPS
            )
            accumulated = accumulated + weight

        displacement = displacement.clamp(-max_displacement, max_displacement)
        orientation_weight = (accumulated / S).clamp(0.0, 1.0)
        cosine = torch.cos(self.thetas)[:, None, None]
        sine = torch.sin(self.thetas)[:, None, None]

        A00 = (orientation_weight * cosine * cosine).sum(0) + 1e-4
        A01 = (orientation_weight * cosine * sine).sum(0)
        A11 = (orientation_weight * sine * sine).sum(0) + 1e-4
        b0 = (orientation_weight * cosine * displacement).sum(0)
        b1 = (orientation_weight * sine * displacement).sum(0)

        det = A00 * A11 - A01 * A01 + EPS
        dx = (b0 * A11 - b1 * A01) / det
        dy = (b1 * A00 - b0 * A01) / det
        limiter = torch.clamp(
            torch.sqrt(dx * dx + dy * dy) / max(max_displacement, 0.25), min=1.0
        )
        dx, dy = dx / limiter, dy / limiter

        conditioning = (det / (A00 * A11 + EPS)).clamp(0.0, 1.0)
        amp_conf = (orientation_weight.sum(0) / O).clamp(0.0, 1.0)
        confidence = conditioning * amp_conf

        projected = dx[None] * cosine + dy[None] * sine
        predicted = (
            TAU * self.freqs[:, None, None, None] * projected[None]
        ).reshape(self.num_bands, H, H)
        predicted = predicted * (confidence[None] * 1.75).clamp(0.0, 1.0)
        return dx, dy, confidence, predicted

    def project_affine_nullspace(self, proposed_delta, source_z, strength: float = 1.0):
        strength = float(np.clip(strength, 0.0, 1.0))
        if strength <= 1e-6:
            return proposed_delta, 0.0

        gx = 1j * self.kx[:, None, None] * source_z
        gy = 1j * self.ky[:, None, None] * source_z
        Xb, Yb = self.X[None], self.Y[None]
        basis = [
            gx,
            gy,
            -Yb * gx + Xb * gy,
            Xb * gx + Yb * gy,
            Xb * gx - Yb * gy,
            Yb * gx + Xb * gy,
        ]

        ortho = []
        for vec in basis:
            residual = vec.clone()
            for q in ortho:
                coeff = torch.real((torch.conj(q) * residual).sum(0))
                residual = residual - q * coeff[None]
            rr = torch.real((torch.conj(residual) * residual).sum(0))
            q = torch.where(
                (rr > 1e-10)[None],
                residual / torch.sqrt(rr.clamp(min=1e-10))[None],
                torch.zeros_like(residual),
            )
            ortho.append(q)

        total = (proposed_delta.abs() ** 2).sum() + EPS
        safe = torch.empty_like(proposed_delta)
        removed = torch.zeros((), device=self.device)
        for c in range(3):
            delta = proposed_delta[c]
            projection = torch.zeros_like(delta)
            residual = delta.clone()
            for q in ortho:
                coeff = torch.real((torch.conj(q) * residual).sum(0))
                comp = q * coeff[None]
                projection = projection + comp
                residual = residual - comp
            safe[c] = delta - strength * projection
            removed = removed + ((strength * projection).abs() ** 2).sum()
        return safe, float((removed / total).clamp(0.0, 1.0).item())


@dataclass
class RailState:
    previous_source_z: Optional[object] = None
    previous_source_low: Optional[object] = None
    previous_source_residual: Optional[object] = None
    target_z: Optional[object] = None
    target_low: Optional[object] = None
    target_residual: Optional[object] = None
    output_z: Optional[object] = None
    removed_fraction: float = 0.0

    def reset(self) -> None:
        self.previous_source_z = None
        self.previous_source_low = None
        self.previous_source_residual = None
        self.target_z = None
        self.target_low = None
        self.target_residual = None
        self.output_z = None
        self.removed_fraction = 0.0


class LayerPhaseRail:
    """Persistent generated appearance moved by the current source frame."""

    def __init__(self, size: int = 128, device: str = "cuda") -> None:
        self.p = TorchGaborPyramid(size=size, orientations=6, device=device)
        self.state = RailState()
        self.target: Optional[np.ndarray] = None
        self.force_refresh = True

    @property
    def device(self) -> str:
        return str(self.p.device)

    def reset(self) -> None:
        self.state.reset()
        self.force_refresh = True

    def set_target(self, target_bgr: np.ndarray) -> None:
        if target_bgr.shape != (self.p.size, self.p.size, 3):
            raise ValueError(
                f"target must be {(self.p.size, self.p.size, 3)}, got {target_bgr.shape}"
            )
        self.target = np.ascontiguousarray(target_bgr.astype(np.float32))
        self.force_refresh = True

    @torch.no_grad()
    def process(
        self,
        source_bgr: np.ndarray,
        *,
        phase_lock: float = 0.92,
        style_strength: float = 1.0,
        nullspace_strength: float = 0.0,
        structure_follow: float = 0.92,
        detail_follow: float = 0.85,
        max_displacement: float = 3.5,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        if self.target is None:
            self.set_target(source_bgr)

        p, st = self.p, self.state
        src = p.to_device(source_bgr)
        # BGR Rec.709 luma. Channel order does not affect phase transport, but
        # using the correct weights improves the motion measurement.
        gray = 0.0722 * src[..., 0] + 0.7152 * src[..., 1] + 0.2126 * src[..., 2]
        source_z, _, _ = p.analyze_gray(gray)
        source_low, source_residual = p.analyze_bgr_low_residual(src)

        if st.previous_source_z is None:
            dx = torch.zeros_like(gray)
            dy = torch.zeros_like(gray)
            confidence = torch.zeros_like(gray)
            predicted = torch.zeros(
                (p.num_bands, p.size, p.size), device=p.device
            )
        else:
            dx, dy, confidence, predicted = p.estimate_bound_phase_flow(
                source_z, st.previous_source_z, max_displacement
            )

        coherence = p.cross_scale_coherence(source_z)
        coherence_map = coherence.max(dim=0).values

        if self.force_refresh or st.target_z is None:
            proposal = p.to_device(self.target)
            source_z_bgr, _, _ = p.analyze_bgr(src)
            prop_z, prop_low, prop_residual = p.analyze_bgr(proposal)
            safe_delta, removed = p.project_affine_nullspace(
                prop_z - source_z_bgr, source_z, nullspace_strength
            )
            st.target_z = source_z_bgr + float(style_strength) * safe_delta
            st.target_low = (
                float(structure_follow) * source_low
                + (1.0 - float(structure_follow)) * prop_low
            )
            st.target_residual = (
                (1.0 - float(detail_follow)) * source_residual
                + float(detail_follow) * prop_residual
            )
            st.output_z = st.target_z.clone()
            st.removed_fraction = removed
            self.force_refresh = False
        else:
            st.target_z = st.target_z.abs() * torch.exp(
                1j * (torch.angle(st.target_z) + predicted[None])
            )
            if st.previous_source_low is not None:
                st.target_low = st.target_low + (source_low - st.previous_source_low)
            if st.previous_source_residual is not None:
                st.target_residual = st.target_residual + float(detail_follow) * (
                    source_residual - st.previous_source_residual
                )

            rail = st.output_z.abs() * torch.exp(
                1j * (torch.angle(st.output_z) + predicted[None])
            )
            lock = (
                float(phase_lock) * (0.30 + 0.70 * coherence)
            ).clamp(0.0, 1.0)
            lock = lock[None].expand(
                p.num_scales, -1, -1, -1
            ).reshape(p.num_bands, p.size, p.size)[None]
            phase_unit = (
                lock * torch.exp(1j * torch.angle(rail))
                + (1.0 - lock) * torch.exp(1j * torch.angle(st.target_z))
            )
            memory = 0.35 + 0.45 * float(phase_lock)
            amplitude = memory * rail.abs() + (1.0 - memory) * st.target_z.abs()
            st.output_z = amplitude * torch.exp(1j * torch.angle(phase_unit))

        output_low = (
            float(structure_follow) * source_low
            + (1.0 - float(structure_follow)) * st.target_low
        )
        output_residual = (
            (1.0 - float(detail_follow)) * source_residual
            + float(detail_follow) * st.target_residual
        )
        output = p.synthesize_bgr(st.output_z, output_low, output_residual)

        metrics = {
            "motion": float(torch.sqrt(dx * dx + dy * dy).mean().item()),
            "confidence": float(confidence.mean().item()),
            "coherence": float(coherence_map.mean().item()),
            "removed": float(st.removed_fraction),
        }
        st.previous_source_z = source_z
        st.previous_source_low = source_low
        st.previous_source_residual = source_residual
        return (
            output.detach().cpu().numpy().astype(np.float32),
            coherence_map.detach().cpu().numpy().astype(np.float32),
            metrics,
        )


def selftest() -> None:
    if not TORCH_OK:
        raise RuntimeError("torch is not installed")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = TorchGaborPyramid(size=64, device=device)
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32)
    image = (
        0.35
        + 0.25 * np.sin(xx * 0.18)
        + 0.18 * np.cos(yy * 0.13)
        + 0.20 * np.exp(-((xx - 32) ** 2 + (yy - 30) ** 2) / 90.0)
    ).astype(np.float32)
    image = np.clip(image, 0, 1)
    z, low, residual = p.analyze_gray(p.to_device(image))
    recon = p.synthesize_gray(z, low, residual).cpu().numpy()
    err = float(np.max(np.abs(recon - image)))
    if err > 8e-5:
        raise AssertionError(f"PhaseRail reconstruction failed: {err}")

    shifted = cv2.warpAffine(
        image,
        np.float32([[1, 0, 1.1], [0, 1, -0.7]]),
        (64, 64),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    z1, _, _ = p.analyze_gray(p.to_device(shifted))
    dx, dy, conf, _ = p.estimate_bound_phase_flow(z1, z, 3.0)
    good = conf > torch.quantile(conf.flatten(), 0.65)
    est_x = float(dx[good].median().item())
    est_y = float(dy[good].median().item())
    if abs(est_x + 1.1) > 0.55 or abs(est_y - 0.7) > 0.55:
        raise AssertionError(f"PhaseRail translation failed: {(est_x, est_y)}")


if __name__ == "__main__":
    selftest()
    print("fx_phase_rail selftest PASS")
