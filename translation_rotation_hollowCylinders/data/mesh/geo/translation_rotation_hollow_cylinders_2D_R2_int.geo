SetFactory("OpenCASCADE");


// === VARIABLES DEFINITION ===
// First cylinder
R_int_1 = 0.0154; // Internal radius
R_ext_1 = 0.0197;
h_1 = 0.04337;
R_1 = 0;
Z_1 = -1e-05;

// Second cylinder
R_int_2 = 0.0304;
R_ext_2 = 0.0346975;
h_2 = 0.07983;

// Internal domain
R_Omega = 0.10579077712636391;

// Mesh caracteristics
Ngamma = 66;
minSize = 0.0005;
maxSize = 0.005;

// === END OF VARIABLES DEFINITION ===


// First cylinder
// Setting the rectangle's points
Point(1) = {R_int_1+R_1, -(h_1/2)+Z_1, 0};
Point(2) = {R_ext_1+R_1, -(h_1/2)+Z_1, 0};
Point(3) = {R_ext_1+R_1, (h_1/2)+Z_1, 0};
Point(4) = {R_int_1+R_1, (h_1/2)+Z_1, 0};

// Joining the points with lines
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Creating the loop and the surface
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Creating the Physical surface from the plane one
Physical Surface("Internal_cylinder", 300) = {1};


// Second cylinder

Point(5) = {R_int_2, -h_2/2, 0};
Point(6) = {R_ext_2, -h_2/2, 0};
Point(7) = {R_ext_2, h_2/2, 0};
Point(8) = {R_int_2, h_2/2, 0};

Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

Curve Loop(2) = {5, 6, 7, 8};
Plane Surface(2) = {2};

Physical Surface("Second_cylinder", 301) = {2};


// Internal domain
// Creating the points that will be used to create the domain
Point(9) = {0, 0, 0};
Point(10) = {0, -R_Omega, 0};
Point(11) = {0, R_Omega, 0};

// Creating the lines of the domain (it's a semi-circle)
Line(9) = {9, 11};
Circle(10) = {11, 9, 10};
Line(11) = {10, 9};

// Line loop and surface
Curve Loop(3) = {9, 10, 11};
Plane Surface(3) = {3};

// Excluding the first and second cylinders from the domain
BooleanDifference { Surface{3}; Delete; } { Surface{1, 2}; }

// Getting physical
Physical Surface("Internal_domain", 302) = {3};
Physical Curve("Internal_boundary", 200) = {11};


// === DEFINING THE MESH ===
// Number of points on the boundary line of the domain
Transfinite Curve(11) = Ngamma Using Progression 1;


// Size fields that determines the mesh
Field[1] = Distance;
Field[1].CurvesList = {1, 2, 3, 4};

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].DistMin = 0;
Field[2].DistMax = 6 * (R_ext_1 - R_int_1);
Field[2].SizeMin = minSize;
Field[2].SizeMax = maxSize / 2;
Field[2].StopAtDistMax = 1;

Field[3] = Threshold;
Field[3].InField = 1;
Field[3].DistMin = 6 * (R_ext_1 - R_int_1);
Field[3].DistMax = R_Omega;
Field[3].SizeMin = maxSize / 2;
Field[3].SizeMax = maxSize;

Field[4] = Distance;
Field[4].CurvesList = {5, 6, 7, 8};

Field[5] = Threshold;
Field[5].InField = 4;
Field[5].DistMin = 0;
Field[5].DistMax = 6 * (R_ext_2 - R_int_2);
Field[5].SizeMin = minSize;
Field[5].SizeMax = maxSize / 2;
Field[5].StopAtDistMax = 1;

Field[6] = Threshold;
Field[6].InField = 4;
Field[6].DistMin = 6 * (R_ext_2 - R_int_2);
Field[6].DistMax = R_Omega;
Field[6].SizeMin = maxSize / 2;
Field[6].SizeMax = maxSize;

Field[7] = Min;
Field[7].FieldsList = {2, 3, 5, 6};

Background Field = 7;
