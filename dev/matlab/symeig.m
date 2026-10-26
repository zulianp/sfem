clear;


syms a00 a01 a02 a11 a12 a22 real
sym d0 real
sym d1 real
sym d2 real

% A = [a00, a01, a02; 
%      a01, a11, a12;
%      a02, a12, a22 ];


% A = [a00, a01, a02; 
%      a01, 0, 0;
%      a02, 0, 0 ];

A = [a00, a01, a02; 
     a01, a11, a12;
     a02, a12, a22 ];

[V,D] = eig(A);

d0=D(1,1)
d1=D(2,2)
d2=D(3,3)

assume(d0, 'real')
assume(d1, 'real')
assume(d2, 'real')

disp(sprintf('D[0]=(%s)', d0));    

disp(' ')
disp(sprintf('D[1]=(%s)', d1));    

disp(' ')
disp(sprintf('D[2]=(%s)', d2));    


disp(V)