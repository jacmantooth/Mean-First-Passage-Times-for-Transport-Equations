function SolveMFPTJumpFiniteDifference

close all;


cord = [[191,11,241,234];[241,234,278,452];[278,452,309,639];[309,639,310,627];[309,627,331,756];[16,237, 279,233];
                 [279,233,1145,249];[1145,249,1406,252];[1406,252, 1469,256];[389,10,296,300];[296,300,270,411];

                 [270,411,149,780];[149,780,98,950];[98,950,22,1140];[11,455,170,453];[170,453,179,447];[179,447,280,451];

                 [15,360,222,550];[222,550,310,639];[310,639,443,745];[878,10,746,717];[746,717,724,828];[724,828,700,911];

                 [949,6,930,48];[930,48,874,181];[855,119,880,243];[880,243,912,396];[912,396,1003,985];[1003,985,1026,1152];

                 [1026,1152,1043,1309];[1043,1309,1050,1467];[519,7,547,116];[547,116,628,560];[628,560,721,1002];[721,1002,744,1141];

                 [744,1141,788,1326];[788,1326,812,1465];[24,795, 330,753];[330,753, 447,747];[447,747, 962,698];[962,698,1184,701];
                 [1184,701,1220,698];[1220,698,1465,667];[1275,6,1291,26];[1291,26,1318,47];[1318,47,1350,69];[1350,69,1390,82];
                 
                 [1390,82,1423,344];[1423,344,1469,666];[923,460,1049,572];[1049,572,1180,702];[1118,84, 1195,534];[1195,534,1279,1134];
                 [1279,1134,1323,1465];[929,12,1219,616];[1219,616,1317,809];[1317,809,1379,958];[1379,958,1463,1134];[450,750,618,930];
                 [618,930,723,1031];[723,1031,837,1149];[837,1149,1001,1314];[843,1470, 907,1401];[907,1401,1270,1065];[1270,1065,1445,888];
                 [1445,888,1468,843];[996,1318,1138,1467];[13,1117,266,1119];[266,1119,369,1121];[369,1121,487,1124];[487,1124,680,1140];
                 [680,1140,831,1148];[831,1148,1184,1147];[1184,1147,1368,1124];[1368,1124,1465,1121];[1287,1203,1345,1243];[1345,1243,1469,1329];
                 [13,1117, 105,1179];[105,1179,452,1386];[485,1315,587,1387];[587,1387,723,1470];[15,1337,327,1316];[327,1316,490,1315];
                 [490,1315,789,1328];[789,1328,978,1334];[978,1334,1071,1335];[1071,1335,1096,1330];[1096,1330,1469,1329];[421,1122,437,1222];
                 [437,1222,463,1468];[98,950,124,980];[124,980,160,995];[160,995, 300,1000];[300,1000,359,1017];[359,1017,378,1038];
                 [378,1038,395,1078];[395,1078,487,1124]];

C = imread('/Users/jacobmantooth/Downloads/wolftrack3.png');

filename = 'data.csv'; % Specify the path to your CSV file
dataTable = readtable(filename);
filename2 = 'data2.csv'; % Specify the path to your CSV file
dataTable2 = readtable(filename2);

filename3 = 'data3.csv'; % Specify the path to your CSV file
dataTable3 = readtable(filename3);

nx =  120; ny = 120; % Number of internal point to solve at in x & y.
xa = 0; xb = 10; ya = 0; yb = 10; % Boundary points.
%look at the error 
x_f = linspace(xa,xb,nx+2); hx = x_f(2)-x_f(1);
y_f = linspace(ya,yb,ny+2); hy = y_f(2)-y_f(1);
[X_f,Y_f] = meshgrid(x_f,y_f);
X = X_f(:); Y = Y_f(:);
eps = .5;
Z_1 = X./eps; Z_2 = Y./eps;
%ktest =exp(Z_1-Z_2);

ktest = 50.*cos(Z_1).^2;

%k = 25*(dist<10);


angle = pi/2 .* ones(size(X,1), 1);

isbdy = find( (X == xa) | (X == xb) | (Y == ya) | (Y == yb) );
isint = setdiff(1:length(X),isbdy);

xtest = X(isint);
q = zeros(size(xtest));

fun = @(z1, z2, x) besseli(2, 50.*cos(z1).^2) ./ besseli(0, 50.*cos(z1).^2);

% Set the tolerance and function evaluation options
RelTol = 1e-8;
AbsTol = 1e-6;


q = arrayfun(@(x) integral2(@(z1, z2) fun(z1, z2, x), 0, xb/eps, 0, yb/eps, 'RelTol', RelTol, 'AbsTol', AbsTol), xtest);


a = besseli(2,ktest)./besseli(0,ktest);
D112 = 1/2-q./(2*xb/eps*xb/eps) + q.*cos(angle(isint)).^2./(xb/eps*xb/eps);
D222 = 1/2-q./(2*xb/eps*xb/eps) + q.*sin(angle(isint)).^2./(xb/eps*xb/eps);
D122 = q.*cos(angle(isint)).*sin(angle(isint))./(xb/eps*xb/eps);
% ap = besseli(1,k)./besseli(0,k);

D11 = 0.5*(1 - a(isint)) + a(isint).*cos(angle(isint)).^2;
D12 = a(isint) .*cos(angle(isint)).*sin(angle(isint));
D22 = 0.5*(1 - a(isint)) + a(isint).*sin(angle(isint)).^2;




M = get_diff_matrices(nx,ny,hx,hy);

D = 100^2;  f = -ones(nx*ny,1);


fhom = -ones(nx*ny,1);

L = spdiags(D11,0,nx*ny,nx*ny)*M.D2XX+ ...
 2*spdiags(D12,0,nx*ny,nx*ny)*M.D2XY + ...
    spdiags(D22,0,nx*ny,nx*ny)*M.D2YY;
%L = M.D2XX +2*M.D2XY +M.D2YY;
Lhom = spdiags(D112,0,nx*ny,nx*ny)*M.D2XX+ ...
 2*spdiags(D122,0,nx*ny,nx*ny)*M.D2XY + ...
    spdiags(D222,0,nx*ny,nx*ny)*M.D2YY;
L = D*L;
Lhom = D*Lhom;

u = L\f;
uhom = Lhom\fhom;
U_fhom = zeros(size(X));
U_fhom(isint) = uhom;
U_fhom = reshape(U_fhom,nx+2,ny+2);

U_f = zeros(size(X));
%U_f(isint) = a(isint);
%u = u/max(u);

U_f(isint) = u;
U_f = reshape(U_f,nx+2,ny+2);
error_matrix = U_f - U_fhom;

figure('color','w','Position',[100 100 1500 500]); % Create a new figure with white background

% First subplot
subplot(1, 3, 1); % 1 row, 2 columns, first subplot
hold on;
contourf(X_f, Y_f, U_f);
title('FDM')
set(gca, 'YDir', 'reverse');

axis square;

axis off;

% Second subplot
subplot(1, 3, 2); % 1 row, 2 columns, second subplot
hold on;
contourf(X_f, Y_f, U_fhom);
title('Homogenized')
set(gca, 'YDir', 'reverse');

axis square;

axis off;


subplot(1, 3, 3); 
hold on;
contourf(X_f, Y_f, error_matrix);
title('Difference')
set(gca, 'YDir', 'reverse');
axis square;
colorbar;
axis off;

hold off;
%for j = 1:size(cord,1)
    %plot([cord(j,1) cord(j,3)] ,[cord(j,2) cord(j,4)],'-r','linewidth',2)
%end
%plot(dataTable.column1, dataTable.column2,'k');
%plot(dataTable2.column1, dataTable2.column2,'m');
%plot(dataTable3.column1, dataTable3.column2,'w');
%plot(800,750,'g.','markersize',32);



%U_sol = interp2(X_f,Y_f,U_f,x0,y0);

% kplot = zeros(size(X_f));
% kplot(isint) = reshape(k,nx,ny);
% 
% figure('color','w');
% pcolor(X_f, Y_f, kplot);
% shading interp;
% colorbar;

% Solve by FEM
return

% time dependent code;

function M = get_diff_matrices(nx,ny,hx,hy)
    
%%Making derivative matrices.

ex = ones(nx,1); ey = ones(ny,1);
Ix = speye(nx); Iy = speye(ny);


bc = 0;

if bc == 1 

%NBC
exupper = ones(nx,1);
exupper(2) = 2;

exlower = ones(ny,1);
exlower(end-1) = 2;

D2Xtest = spdiags([exlower -2*ex exupper],-1:1,nx,nx)/(hx*hx);
M.D2XX = kron(D2Xtest,Iy);


eyupper = ones(nx,1);
eylower = ones(ny,1);

D2Ytest = spdiags([eylower -2*ey eyupper],-1:1,nx,nx)/(hy*hy);
M.D2YY = kron(Ix,D2Ytest);

ns = size(M.D2YY, 1);

for n = 1:nx
    if n < ns
        if n + 1 <= ns
            M.D2YY(n, n+1) = 2/(hy*hy); 
        end
        if n - 1 >= 1
            M.D2YY(n, n-1) = 0; 
        end
    end
end

for n = ns-(nx-1):ns
    if n <= ns
        if n + 1 <= ns
            M.D2YY(n, n+1) = 0;
        end
        if n - 1 >= 1
            M.D2YY(n, n-1) = 2/(hy*hy);
        end
    end
end

% Create vectors of ones
exmidupper = ones(nx,1); 
 exmidupper(2) = 0;
exmidlower = ones(nx,1); 
 exmidlower(end-1) = 0;

exmidupper2 = ones(nx,1); 
exmidupper2(2) = 0;
exmidlower2 = ones(nx,1);  
exmidlower2(end -1 ) = 0;
% Create the sparse matrices
DCXmid = spdiags([-exmidlower 0*ex exmidupper],-1:1,nx,nx)/(2*hx);
DCYmid = spdiags([-exmidlower2 0*ex exmidupper2],-1:1,nx,nx)/(2*hx);

% Create the Kronecker product
M.D2XY = kron(DCXmid,DCYmid);

%spy(M.D2XY)

else
%keep this DBC
DCX = spdiags([-ex 0*ex ex],-1:1,nx,nx)/(2*hx);
DCY = spdiags([-ey 0*ey ey],-1:1,ny,ny)/(2*hy);

D2X = spdiags([ex -2*ex ex],-1:1,nx,nx)/(hx*hx);
D2Y = spdiags([ey -2*ey ey],-1:1,ny,ny)/(hy*hy);

M.D2XX = kron(D2X,Iy);
M.D2YY = kron(Ix,D2Y);
M.D2XY = kron(DCX,DCY);
M.DCX = kron(DCX,Iy); M.DCY = kron(Ix,DCY);


end 

