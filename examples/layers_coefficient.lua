-- ============================================================================
-- Configuration parameters for layered diffusion problem
-- ============================================================================

-- Layer orientation: angle in degrees from horizontal (0° = horizontal, 90° = vertical)
LAYER_ANGLE = 12

-- Number of layers
NUM_LAYERS = 8

-- Thickness of each high-conductivity layer
LAYER_THICKNESS = 0.04

-- Spacing between layers (center to center distance)
LAYER_SPACING = 0.1

-- Coefficient value in high-conductivity layers
LAYER_VALUE = 1e6

-- Background coefficient value
BACKGROUND_VALUE = 1.0

-- Margin from boundary to keep layers away from Dirichlet boundaries
BOUNDARY_MARGIN = 0.08

-- ============================================================================
-- Problem definition
-- ============================================================================

-- Dirichlet boundary condition
function g(x, y) 
   -- Simple linear variation for testing
   return 1-x
end

-- Right-hand side / source term
function f(x, y)
   -- Constant source term
   return 0.0
end

-- Diffusion coefficient (defines the layered structure)
function alpha(x, y)
   -- Convert angle to radians
   local angle_rad = LAYER_ANGLE * math.pi / 180.0
   
   -- Compute coordinate along the normal direction to the layers
   -- For angle=0° (horizontal layers): coord ≈ y
   -- For angle=90° (vertical layers): coord ≈ x
   local coord = math.cos(angle_rad) * y + math.sin(angle_rad) * x
   
   -- Compute the perpendicular distances to boundaries
   -- This ensures layers don't touch Dirichlet boundaries
   local perp_to_left = math.cos(angle_rad) * x - math.sin(angle_rad) * y
   local perp_to_right = math.cos(angle_rad) * (1.0 - x) - math.sin(angle_rad) * (1.0 - y)
   local perp_to_bottom = math.cos(angle_rad) * y + math.sin(angle_rad) * x
   local perp_to_top = math.cos(angle_rad) * (1.0 - y) + math.sin(angle_rad) * (1.0 - x)
   
   -- Check if we're too close to any boundary
   local min_dist = math.min(
      math.abs(x),           -- distance to left
      math.abs(1.0 - x),     -- distance to right
      math.abs(y),           -- distance to bottom
      math.abs(1.0 - y)      -- distance to top
   )
   
   if min_dist < BOUNDARY_MARGIN then
      return BACKGROUND_VALUE
   end
   
   -- Determine if point is inside a layer
   -- Center the layers in the valid region
   local valid_range = 1.0 - 2.0 * BOUNDARY_MARGIN
   local start_coord = BOUNDARY_MARGIN + (valid_range - (NUM_LAYERS - 1) * LAYER_SPACING) / 2.0
   
   for i = 0, NUM_LAYERS - 1 do
      local layer_center = start_coord + i * LAYER_SPACING
      local dist_to_center = math.abs(coord - layer_center)
      
      if dist_to_center < LAYER_THICKNESS / 2.0 then
         return LAYER_VALUE
      end
   end
   
   return BACKGROUND_VALUE
end

-- Dirichlet boundary specification
function is_dirichlet(x, y)
   -- Dirichlet on left and right boundaries (x = 0 and x = 1)
   -- Neumann on top and bottom boundaries
   if x < 1e-6 then return true end
   if x > 1.0 - 1e-6 then return true end
   return false
end
