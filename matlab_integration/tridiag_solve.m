function x = tridiag_solve(d,u,l)
    n = length(d);
    Bin = [[l'; 0], d', [0; u']];
    A = spdiags(Bin, [-1, 0, 1], n, n);
    x = A \ eye(n, 1);
