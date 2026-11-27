// unstructured triangle mesh of the unit square; h = mesh size
If(!Exists(h)) h = 0.0625; EndIf
Point(1) = {0,0,0,h}; Point(2) = {1,0,0,h}; Point(3) = {1,1,0,h}; Point(4) = {0,1,0,h};
Line(1) = {1,2}; Line(2) = {2,3}; Line(3) = {3,4}; Line(4) = {4,1};
Curve Loop(1) = {1,2,3,4}; Plane Surface(1) = {1};
