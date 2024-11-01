function [C, S] = tridiag_wkm(l, d, u)
    n = length(d);
    Bin = [[l'; 0], d', [0; u']];
    A = spdiags(Bin, [-1, 0, 1], n, n);
    % wkm evaluates COSH(SQRT(A)) and SINHC(SQRT(A)),
    % further there are the identities SQRT(-A) = i SQRT(A) and COSH(i*A) = COS(A).
    [C, S] = wkm(-A);
