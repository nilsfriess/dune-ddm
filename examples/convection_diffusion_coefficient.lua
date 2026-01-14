function alpha(x, y)
   nx = 8
   ny = 8

   a1 = 1e-6
   a2 = 1

   ix = math.floor(x * nx)
   iy = math.floor(y * ny)

   coeff = a1
   if (ix % 2 == iy % 2) then
      coeff = a2
   end
   
   return coeff
end

function f(x, y)
   return 0.
end

function is_dirichlet(x, y)
   return x < 1e-6 or y < 1e-6
end

function b1(x, y)
   return 1. / 3
end

function b2(x, y)
   return 1
end

function g(x, y)
   if (x < 1e-6) then
      return 1.
   else
      return 0.
   end   
end
