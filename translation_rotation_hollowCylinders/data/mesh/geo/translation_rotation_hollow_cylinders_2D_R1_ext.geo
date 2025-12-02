// Gmsh project created on Wed Nov  5 23:07:29 2025
SetFactory("OpenCASCADE");


// === VARIABLES DEFINITION ===
// External domain
R_Omega = 0.10579077712636391;

// Mesh caracteristics
Ngamma = 66;
minSize = 0.0003;
maxSize = 0.005;

// === END OF VARIABLES DECLARATION ===


// External domain
// Defining the points of the domain
Point(1) = {0, 0, 0};
Point(2) = {0, -R_Omega, 0};
Point(3) = {0, R_Omega, 0};

// Creating the semi-circle
Line(1) = {1, 3};
Circle(2) = {3, 1, 2};
Line(3) = {2, 1};

// Creating the surface
Curve Loop(1) = {1, 2, 3};
Plane Surface(1) = {1};

Point{1} In Surface{1};

// Creating the physical things
Physical Surface("External_domain", 303) = {1};
Physical Curve("External_boundary", 201) = {2};
Physical Point(0) = {1};


// === DEFINING THE MESH ===
// Number of points on the boundary line of the domain
Transfinite Curve(2) = Ngamma Using Progression 1;

Field[1] = Distance;
Field[1].PointsList = {1};

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].DistMin = 0;
Field[2].DistMax = R_Omega;
Field[2].SizeMin = minSize;
Field[2].SizeMax = maxSize;

Background Field = 2;
