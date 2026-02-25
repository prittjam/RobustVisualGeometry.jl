# =============================================================================
# Conic Fitting — Geometric Distance (Levenberg-Marquardt)
# =============================================================================

"""
    _closest_point_on_ellipse(cx, cy, a, b, phi, px, py) -> (cpx, cpy)

Find the closest point on an ellipse to (px, py) using the Eberly bisection
algorithm (robust, guaranteed convergence).

Reference: D. Eberly, "Distance from a Point to an Ellipse, an Ellipsoid, or a
Hyperellipsoid", Geometric Tools.
"""
function _closest_point_on_ellipse(cx, cy, a, b, phi, px, py)
    cos_phi, sin_phi = cos(phi), sin(phi)
    dx = px - cx
    dy = py - cy
    u = cos_phi * dx + sin_phi * dy
    v = -sin_phi * dx + cos_phi * dy

    # Ensure e0 >= e1 (largest semi-axis first)
    if a >= b
        e0, e1, y0, y1 = a, b, abs(u), abs(v)
    else
        e0, e1, y0, y1 = b, a, abs(v), abs(u)
    end

    x0, x1 = _eberly_bisect_2d(e0, e1, y0, y1)

    # Restore signs
    if a >= b
        x_local = copysign(x0, u)
        y_local = copysign(x1, v)
    else
        x_local = copysign(x1, u)
        y_local = copysign(x0, v)
    end

    cpx = cx + cos_phi * x_local - sin_phi * y_local
    cpy = cy + sin_phi * x_local + cos_phi * y_local
    cpx, cpy
end

"""
    _eberly_bisect_2d(e0, e1, y0, y1) -> (x0, x1)

Eberly's bisection for point-to-ellipse distance in axis-aligned first-quadrant
coordinates. e0 >= e1 > 0, y0 >= 0, y1 >= 0.
"""
function _eberly_bisect_2d(e0, e1, y0, y1)
    if y1 > 0
        if y0 > 0
            # General case: both coordinates nonzero
            z0 = y0 / e0
            z1 = y1 / e1
            g = z0^2 + z1^2 - 1.0
            if g != 0.0
                r0 = (e0 / e1)^2
                sbar = _eberly_find_root(r0, z0, z1, g)
                x0 = r0 * y0 / (sbar + r0)
                x1 = y1 / (sbar + 1.0)
            else
                x0 = y0
                x1 = y1
            end
        else
            # y0 == 0: closest point is on the y-axis portion of the ellipse
            x0 = 0.0
            x1 = e1
        end
    else
        # y1 == 0: closest point is on the x-axis
        numer0 = e0 * y0
        denom0 = e0^2 - e1^2
        if numer0 < denom0
            xde0 = numer0 / denom0
            x0 = e0 * xde0
            x1 = e1 * sqrt(max(0.0, 1.0 - xde0^2))
        else
            x0 = e0
            x1 = 0.0
        end
    end
    x0, x1
end

"""
    _eberly_find_root(r0, z0, z1, g) -> s

Find the unique root s > -1 of
    g(s) = (r0*z0/(s+r0))^2 + (z1/(s+1))^2 - 1 = 0
via bisection, where r0 = (e0/e1)^2 >= 1.
"""
function _eberly_find_root(r0, z0, z1, g)
    n0 = r0 * z0
    s0 = z1 - 1.0
    if g < 0.0
        s1 = 0.0
    else
        s1 = sqrt(n0^2 + z1^2) - 1.0
    end
    s = 0.0
    for _ in 1:100
        s = 0.5 * (s0 + s1)
        (s == s0 || s == s1) && break
        g = (n0 / (s + r0))^2 + (z1 / (s + 1.0))^2 - 1.0
        if g > 0.0
            s0 = s
        elseif g < 0.0
            s1 = s
        else
            break
        end
    end
    s
end

@inline _ellipse_params_from_5(p) = (p[1], p[2], abs(p[3]), abs(p[4]), p[5])

"""
    _finalize_geometric(p, pts, sigma, stop_reason, iter) -> ConicFitResult

Convert 5-param ellipse vector to conic θ, compute Sampson residuals, package result.
Used by geometric distance fitting exit paths.
"""
function _finalize_geometric(p, pts::Vector{SVector{2,Float64}}, sigma::Real,
                             stop_reason::Symbol, iter::Int)
    cx, cy, a, b, phi = _ellipse_params_from_5(p)
    ell = Ellipse(Point2(cx, cy), a, b, phi)
    Q = HomEllipseMat(ell)
    theta = SVector{6,Float64}(Q[1,1], 2*Q[1,2], Q[2,2], 2*Q[1,3], 2*Q[2,3], Q[3,3])
    theta = theta / norm(theta)
    xis = _build_carriers(pts)
    Lambdas = _build_covariances(pts, sigma^2)
    residuals = _compute_sampson_residuals(sampson_distance, theta, xis, Lambdas)
    n = length(pts)
    Attributed(theta, RobustAttributes(stop_reason, residuals, ones(n), Float64(sigma), iter))
end

"""
    fit_conic_geometric(points; sigma=1.0, max_iter=100, rtol=1e-10) -> ConicFitResult

Levenberg-Marquardt minimization of geometric (orthogonal) distance.
Gives the ML estimate under isotropic Gaussian noise but is expensive
and not robust to outliers.

# Arguments
- `points`: 2×N matrix, N×2 matrix, or Vector of 2D points
- `sigma::Real=1.0`: Noise standard deviation
- `max_iter::Int=100`: Maximum LM iterations
- `rtol::Float64=1e-10`: Convergence tolerance
"""
function fit_conic_geometric(points; sigma::Real=1.0, max_iter::Int=100, rtol::Float64=1e-10)
    pts = _prepare_points(points)
    n = length(pts)

    # Initialize from FNS
    result0 = fit_conic_fns(points; sigma=sigma)
    theta0 = result0.value

    local e0
    try
        e0 = conic_to_ellipse(theta0)
    catch
        cx = sum(p -> p[1], pts) / n
        cy = sum(p -> p[2], pts) / n
        r = sqrt(sum(p -> (p[1]-cx)^2 + (p[2]-cy)^2, pts) / n)
        e0 = Ellipse(Point2(cx, cy), r, r, 0.0)
    end
    p = [Float64(e0.center[1]), Float64(e0.center[2]),
         Float64(e0.a), Float64(e0.b), Float64(e0.θ)]

    lambda = 1e-3
    nu = 2.0

    function compute_geo_residuals(p)
        cx, cy, a, b, phi = _ellipse_params_from_5(p)
        r = Vector{Float64}(undef, n)
        @inbounds for i in 1:n
            cpx, cpy = _closest_point_on_ellipse(cx, cy, a, b, phi, pts[i][1], pts[i][2])
            r[i] = sqrt((pts[i][1] - cpx)^2 + (pts[i][2] - cpy)^2)
            dx_c = pts[i][1] - cx
            dy_c = pts[i][2] - cy
            cos_phi, sin_phi = cos(phi), sin(phi)
            u = cos_phi * dx_c + sin_phi * dy_c
            v = -sin_phi * dx_c + cos_phi * dy_c
            if (u/max(a,eps()))^2 + (v/max(b,eps()))^2 < 1.0
                r[i] = -r[i]
            end
        end
        r
    end

    function compute_jacobian(p, r)
        J = Matrix{Float64}(undef, n, 5)
        dp = 1e-7
        for j in 1:5
            p_plus = copy(p)
            p_plus[j] += dp
            r_plus = compute_geo_residuals(p_plus)
            J[:, j] = (r_plus - r) / dp
        end
        J
    end

    r = compute_geo_residuals(p)
    cost = sum(abs2, r)

    for iter in 1:max_iter
        J = compute_jacobian(p, r)
        JtJ = J' * J
        Jtr = J' * r

        H = JtJ + lambda * Diagonal(max.(diag(JtJ), 1e-10))
        dp = -(H \ Jtr)

        p_new = p + dp
        r_new = compute_geo_residuals(p_new)
        cost_new = sum(abs2, r_new)

        if cost_new < cost
            p = p_new
            r = r_new
            cost = cost_new
            lambda = max(lambda / 3.0, 1e-10)
            nu = 2.0
        else
            lambda = min(lambda * nu, 1e10)
            nu *= 2.0
        end

        rel_change = norm(dp) / (norm(p) + 1e-10)
        if rel_change < rtol
            return _finalize_geometric(p, pts, sigma, :converged, iter)
        end
    end

    _finalize_geometric(p, pts, sigma, :max_iterations, max_iter)
end
