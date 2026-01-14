function g(x, y) 
   return 1.0 - x
end

function f(x, y)
   return 0
end

function alpha(x, y)
   local kappa = 1.0
   
   -- Diagonal band with variable coefficient
   if x > 0.3 and x < 0.9 and y > 0.6 - (x - 0.3) / 6 and y < 0.8 - (x - 0.3) / 6 then
      kappa = 1e5 * (x + y) * 10.0
   end
   
   -- Lower left triangular region with variable coefficient
   if x > 0.1 and x < 0.5 and y > 0.1 + x and y < 0.25 + x then
      kappa = 1e5 * (1.0 + 7.0 * y)
   end
   
   -- Lower right diagonal band with constant high coefficient
   if x > 0.5 and x < 0.9 and y > 0.15 - (x - 0.5) * 0.25 and y < 0.35 - (x - 0.5) * 0.25 then
      kappa = 1e5 * 2.5
   end
   
   -- Checkerboard pattern with variable coefficient
   local ix = math.floor(15.0 * x)
   local iy = math.floor(15.0 * y)
   if ix % 2 == 0 and iy % 2 == 0 then
      kappa = 1e5 * (1.0 + ix + iy)
   end
   
   return kappa
end

function is_dirichlet(x, y)
   -- Dirichlet on left and right boundaries (x = 0 and x = 1)
   -- Neumann on top and bottom boundaries
   if x < 1e-6 then return true end
   if x > 1.0 - 1e-6 then return true end
   return false
end
