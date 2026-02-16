-- ============================================================================
-- Configuration parameters for 3D layered diffusion problem
-- ============================================================================

-- Layer orientation using two angles:
-- LAYER_ANGLE_XY: Azimuthal angle in xy-plane (0° = x-direction, 90° = y-direction)
-- LAYER_TILT: Tilt angle from vertical (0° = horizontal layers, 90° = vertical layers)
LAYER_ANGLE_XY = 15   -- degrees
LAYER_TILT = 75       -- degrees

-- Number of layers
NUM_LAYERS = 20

-- Thickness of each high-conductivity layer
LAYER_THICKNESS = 0.02

-- Spacing between layers (center to center distance)
LAYER_SPACING = 0.08

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
function g(x, y, z) 
   -- Simple linear variation for testing
   return 1 - x
end

-- Right-hand side / source term
function f(x, y, z)
   -- Constant source term
   return 0.0
end

-- Diffusion coefficient (defines the 3D layered structure)
function alpha(x, y, z)
   -- Convert angles to radians
   local angle_xy_rad = LAYER_ANGLE_XY * math.pi / 180.0
   local tilt_rad = LAYER_TILT * math.pi / 180.0
   
   -- Compute normal vector to the layers using spherical coordinates
   -- tilt_rad=0 gives (0,0,1) for horizontal layers
   -- tilt_rad=90° gives vertical layers
   local nx = math.sin(tilt_rad) * math.cos(angle_xy_rad)
   local ny = math.sin(tilt_rad) * math.sin(angle_xy_rad)
   local nz = math.cos(tilt_rad)
   
   -- Compute coordinate along the normal direction to the layers
   local coord = nx * x + ny * y + nz * z
   
   -- Check if we're too close to Dirichlet boundaries (x = 0 and x = 1)
   local min_dist = math.min(
      math.abs(x),           -- distance to left (x=0)
      math.abs(1.0 - x)      -- distance to right (x=1)
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
function is_dirichlet(x, y, z)
   -- Dirichlet on left and right boundaries (x = 0 and x = 1)
   -- Neumann on all other boundaries
   if x < 1e-6 then return true end
   if x > 1.0 - 1e-6 then return true end
   return false
end
