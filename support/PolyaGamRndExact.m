% Description: Generate the Polya Gamma distribution using the exact method
function x = PolyaGamRndExact(n,z)
% n: K*1 vector; z: K*1 vector
K = length(n);
x = zeros(K,1);
for i = 1:K
    x(i) = 0;
    for j = 1:n(i)
        x(i) = x(i) + rpg(z(i));
    end;
end;
end

function x = rpg(z)
z = abs(z)*0.5;
fz = pi^2/8+z^2/2;
while (1)
    if rand < mass_texpon(z)
        x = 0.64 + exprnd(1)/fz;
    else
        x = rtigauss(z);
    end;
    S = a_coef(0,x);
    Y = rand*S;
    n = 0;
    while (1)
        n = n+1;
        if mod(n,2)==1
            S = S-a_coef(n,x);
            if (Y<=S), break;
            end;
        else
            S = S+a_coef(n,x);
            if (Y>S), break;
            end;
        end;
    end;
    if (Y<=S), break
    end;
end;
x = x/4;
end

function a = a_coef(n,x)

if (x>0.64)
    a = pi*(n+0.5)*exp(-(n+0.5)^2*pi^2*x/2);
else
    a = (2/pi/x)^1.5*pi*(n+0.5)*exp(-2*(n+0.5)^2/x);
end;
end

function x = rtigauss(z)
R = 0.64; z = abs(z); mu = 1/z; x = R+1;
if (mu > R)
    alpha = 0;
    while rand > alpha
        E = exprnd(1,[1,2]);
        while E(1)^2 > 2*E(2)/R
            E = exprnd(1,[1,2]);
        end;
        x = R/(1+R*E(1))^2;
        alpha = exp(-0.5*z^2*x);
    end;
else
    while x > R
        lambda = 1;
        Y = randn^2;
        x = mu + 0.5*mu^2/lambda*Y - ...
            0.5*mu/lambda*sqrt(4*mu*lambda*Y+(mu*Y)^2);
        if (rand > mu/(mu+x))
            x = mu^2/x;
        end;
    end;
end;
end

function val = mass_texpon(z)
x = 0.64;
fz = pi^2/8+z^2/2;
b = sqrt(1/x)*(x*z-1);
a = -sqrt(1/x)*(x*z+1);

x0 = log(fz)+fz*0.64;
xb = x0-z+log(normpdf(b));
xa = x0+z+log(normpdf(a));

qdivp = 4/pi*(exp(xb)+exp(xa));
val = 1/(1+qdivp);
end
        
                
            
