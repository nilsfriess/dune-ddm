-- Steel properties
STEEL_YOUNGS_MODULUS = 2e11
STEEL_POISSON_RATIO = 0.3

-- Rubber properties
RUBBER_YOUNGS_MODULUS = 2e7
RUBBER_POISSON_RATIO = 0.45

-- Steel reinforcement bars configuration
-- Grid dimensions: x ∈ [0,10], y ∈ [0,1], z ∈ [0,1.5]
BAR_RADIUS = 0.04           -- Radius of each cylindrical steel bar
BAR_START_X = 0.0           -- Where bars start along x-axis
BAR_END_X = 3.0            -- Where bars end along x-axis

-- 2x4 pattern of bars in the y-z cross-section
-- 2 bars in y-direction, 4 bars in z-direction
BAR_POSITIONS_Y = {0.25, 0.75}                    -- y-coordinates of bar centers
BAR_POSITIONS_Z = {0.3, 0.6, 0.9, 1.2}           -- z-coordinates of bar centers

-- Helper function to check if point is inside any steel bar
function is_inside_steel_bar(x, y, z)
   -- Check if x is within the bar range
   if x < BAR_START_X or x > BAR_END_X then
      return false
   end
   
   -- Check if point is within radius of any bar center
   for _, bar_y in ipairs(BAR_POSITIONS_Y) do
      for _, bar_z in ipairs(BAR_POSITIONS_Z) do
         local dist_yz = math.sqrt((y - bar_y)^2 + (z - bar_z)^2)
         if dist_yz <= BAR_RADIUS then
            return true
         end
      end
   end
   
   return false
end

function is_dirichlet(x, y, z)
   return x < 1e-9
end

function youngs_modulus(x, y, z)
   if is_inside_steel_bar(x, y, z) then
      return STEEL_YOUNGS_MODULUS
   else
      return RUBBER_YOUNGS_MODULUS
   end
end

function poisson_ratio(x, y, z)
   if is_inside_steel_bar(x, y, z) then
      return STEEL_POISSON_RATIO
   else
      return RUBBER_POISSON_RATIO
   end
end

function lambda(x, y, z)
   E  = youngs_modulus(x, y, z)
   nu = poisson_ratio(x, y, z)
   
   return E * nu / (1. + nu) / (1. - 2. * nu)
end

function mu(x, y, z)
   E  = youngs_modulus(x, y, z)
   nu = poisson_ratio(x, y, z)

   return E / 2 / (1. + nu)
end
     
