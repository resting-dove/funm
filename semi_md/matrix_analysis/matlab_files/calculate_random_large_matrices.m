calculate_random_large_matrices_(200);

function calculate_random_large_matrices_(size)

A = sprand(size, size, size/ size^2);
A = A + A';
w = eig(A);
A = -(A - diag(sum(A, 1)) - diag(sum(A, 2)));
w = eig(A);
compute_and_display(A, sprintf('randomSparse%d.mat', size));

end

function [fcA, fsA] = wkm_vpa (A)
  fc =@(x) cosh(sqrt(x));


  sinhc =@(x) sinh(x)./x;
  fs =@(x) sinhc(sqrt(x));

  d_vpa = 250; % set digits to use in variable precision arithmetic (vpa)
  hf_vpa =@(s,d) vpa(sym(s,'f'),d); %copied from Nick's myvpa.m; cf. help sym
  d_old = digits(); digits(d_vpa);
  
  % E.B.Davies trick: ensures eigenvalues are distinct; diagonalisation possible
  del_A = randn(length(A)); % random perturbation
  del_A = 10^(-d_vpa/2)*del_A/norm(del_A,1); % make it of norm half of vpa
  [V, D] = eig(hf_vpa(A, d_vpa) + del_A); % E.B.Davies's trick
  
  save(sprintf('randomSparse%deigs.mat', length(A)), 'D', 'V');

  fcA = double(V*diag(fc(diag(D)))/V);
  fsA = double(V*diag(fs(diag(D)))/V);
  digits(d_old);
end

function fsA = fsA_vpa (A)
  sinhc =@(x) sinh(x)./x;
  fs =@(x) sinhc(sqrt(x));

  d_vpa = 250; % set digits to use in variable precision arithmetic (vpa)
  hf_vpa =@(s,d) vpa(sym(s,'f'),d); %copied from Nick's myvpa.m; cf. help sym
  d_old = digits(); digits(d_vpa);
  
  % E.B.Davies trick: ensures eigenvalues are distinct; diagonalisation possible
  del_A = randn(length(A)); % random perturbation
  del_A = 10^(-d_vpa/2)*del_A/norm(del_A,1); % make it of norm half of vpa 
  [V, D] = eig(hf_vpa(A, d_vpa) + del_A); % E.B.Davies's trick

  fsA = double(V*diag(fs(diag(D)))/V);
  digits(d_old);
end

function fcA = fcA_d (A)
  fcA = wkm(A);
end

function fsA = fsA_d (A)
  [~, fsA] = wkm(A);
end


function compute_and_display (A, filename)

  % Compute the fc(A) and fs(A) in vpa to compute approximation errors.
  tic;
  [fcA, fsA] = wkm_vpa(A);
  t_vpa = toc;

  condc = funm_condest1(A,@fcA_d);  % Remember to postmultiply with eps later.
  conds = funm_condest1(A,@fsA_d);

  save(filename, 'A', 'fcA', 'fsA', 'condc', 'conds');
  
  % Compute the fc(A) and fs(A) using Nadukandi--Higham algorithm.
  tic;
  [rcA, rsA] = wkm(A);
  t_apx = toc;
  err_cA = norm(rcA-fcA,1)/norm(fcA,1);
  err_sA = norm(rsA-fsA,1)/norm(fsA,1);
  
  % Display results
  disp(['err_cA = ',num2str(err_cA, '%1.2e'),', ',...
        'err_sA = ',num2str(err_sA, '%1.2e'),', ',...
        ]);

end

function [c, est] = fcA_cond(A, fcA)
factor = norm(A,1)/norm(fcA,1);

[est,v,w,iter] = normest1(@afun);
c = est*factor;
end

function [c,est] = funm_condest1(A,fun,fun_frechet,flag1,varargin)
%FUNM_CONDEST1  Estimate of 1-norm condition number of matrix function.
%    C = FUNM_CONDEST1(A,FUN,FUN_FRECHET,FLAG) produces an estimate of
%    the 1-norm relative condition number of function FUN at the matrix A.
%    FUN and FUN_FRECHET are function handles:
%      - FUN(A) evaluates the function at the matrix A.
%      - If FLAG == 0 (default)
%           FUN_FRECHET(B,E) evaluates the Frechet derivative at B
%              in the direction E;
%        if FLAG == 1
%           - FUN_FRECHET('notransp',E) evaluates the
%                    Frechet derivative at A in the direction E.
%           - FUN_FRECHET('transp',E) evaluates the
%                    Frechet derivative at A' in the direction E.
%    If FUN_FRECHET == @CS then the Frechet derivative is approximated
%    by the complex-step approximation (assuming A is real),
%    while if FUN_FRECHET == @FD then finite differences are used, the latter
%    being the default if FUN_FRECHET is empty.
%    More reliable results are obtained when FUN_FRECHET is supplied.
%    MATLAB'S NORMEST1 (block 1-norm power method) is used, with a random
%    starting matrix, so the approximation can be different each time.
%    C = FUNM_CONDEST1(A,FUN,FUN_FRECHET,FLAG,P1,P2,...) passes extra inputs
%    P1,P2,... to FUN and FUN_FRECHET.
%    [C,EST] = FUNM_CONDEST1(A,...) also returns an estimate EST of the
%    1-norm of the Frechet derivative.
%    Note: this function makes an assumption on the adjoint of the
%    Frechet derivative that, for f having a power series expansion,
%    is equivalent to the series having real coefficients.

fte_diff = 0; cstep = 0;
if nargin < 3 || isempty(fun_frechet)
   fte_diff = 1;
elseif isequal(fun_frechet,@CS)
   cstep = 1;
elseif isequal(fun_frechet,@FD)
   fte_diff = 1;
end

if nargin < 4 || isempty(flag1), flag1 = 0; end

n = length(A);
funA = feval(fun,A,varargin{:});
if fte_diff, tol = eps; end
if cstep, tol = eps^2; end

factor = norm(A,1)/norm(funA,1);

[est,v,w,iter] = normest1(@afun);
c = est*factor;
% 'frechet'
% est
% 'norm A'
% norm(A, 1)
% 'norm fA'
% norm(funA, 1)
% 'factor'
% factor

       %%%%%%%%%%%%%%%%%%%%%%%%% Nested function.
       function Z = afun(flag,X)
       %AFUN  Function to evaluate matrix products needed by NORMEST1.

       if isequal(flag,'dim')
          Z = n^2;
       elseif isequal(flag,'real')
          Z = isreal(A);
       else

          [p,q] = size(X); i = sqrt(-1);
          if p ~= n^2, error('Dimension mismatch'), end
          E = zeros(n);
          Z = zeros(n^2,q);
          for j=1:q

              E(:) = X(:,j);

              if isequal(flag,'notransp')

                 if fte_diff
                    t = sqrt(tol*norm(funA,1))/norm(E,1);
                    Y = (feval(fun,A+t*E,varargin{:}) - funA)/t;
                 elseif cstep
                    t = tol*norm(A,1)/norm(E,1);
                    Y = imag(feval(fun,A+t*i*E,varargin{:}))/t;
                 else
                    if flag1
                       Y = feval(fun_frechet,'notransp',E,varargin{:});
                    else
                       Y = feval(fun_frechet,A,E,varargin{:});
                    end
                 end

              elseif isequal(flag,'transp')

                 if fte_diff
                    t = sqrt(tol*norm(funA,1))/norm(E,1);
                    Y = (feval(fun,A'+t*E,varargin{:}) - funA')/t;
                 elseif cstep
                    t = tol*norm(A,1)/norm(E,1);
                    Y = imag(feval(fun,A'+t*i*E,varargin{:}))/t;
                 else
                    if flag1
                       Y = feval(fun_frechet,'transp',E,varargin{:});
                    else
                       Y = feval(fun_frechet,A',E,varargin{:});
                    end
                 end

              end

              Z(:,j) = Y(:);
          end

       end

       end

end