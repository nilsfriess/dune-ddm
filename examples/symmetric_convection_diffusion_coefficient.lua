function alpha(x, y)
   return 1
end

function f(x, y)
   return 1
end

function is_dirichlet(x, y)
   return (math.abs(x) < 1e-6) or (math.abs(1 - y) < 1e-6)
end

function g(x, y)
   return 0
end
