SetFactory("OpenCASCADE");

// Setting variables
// First cylinder (the one that can move)
R_int_1 = 0.0154; // internal radius of the first cylinder
R_ext_1 = 0.0197; // external radius of the first cylinder
h_1 = 0.04337; // height of the first cylinder
Z_1 = -1e-05; // vertical displacement of the first cylinder
R_1 = -8e-06; // radial displacement of the first cylinder
R_int_2 = 0.0304; // internal radius of the second cylinder
R_ext_2 = 0.0346975; // external radius of the second cylinder
h_2 = 0.07983; // height of the second cylinder
R_Omega = 0.10577568210604932; // radius of the internal domain
Ngamma = 474; // number of nodes on the domain boundary
minSize = 0.0001; // size of the elements inside the spheres
maxSize = 0.0007; // size of the elements far from the spheres
d = 4.001; // distance between the centres of the spheres

// First cylinder
Point(1) = {R_int_1+R_1, -h_1/2 + Z_1, 0};
Point(2) = {R_ext_1+R_1, -h_1/2 + Z_1, 0};
Point(3) = {R_ext_1+R_1, h_1/2 + Z_1, 0};
Point(4) = {R_int_1+R_1, h_1/2 + Z_1, 0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Physical Surface("Hollow_cylinder_1", 300) = {1};


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

Physical Surface("Hollow_cylinder_2", 301) = {2};


// Domain sphere
Point(9) = {0, 0, 0};
Point(10) = {0, -R_Omega, 0};
Point(11) = {0, R_Omega, 0};

Circle(9) = {11, 9, 10};
Line(10) = {11, 9};
Line(11) = {9, 10};

Curve Loop(3) = {-9, 10, 11};
Plane Surface(3) = {3};

BooleanDifference{Surface{3}; Delete;}{Surface{1, 2};}

Physical Surface("Internal_domain", 302) = {3};
Physical Curve("Internal_boundary", 200) = {9};


// Mesh size
Transfinite Curve{9} = Ngamma Using Progression 1;


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
Field[3].StopAtDistMax = 0;

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
Field[6].StopAtDistMax = 0;

Field[7] = Constant;
Field[7].SurfacesList = {3};
Field[7].VIn = maxSize;

Field[8] = Min;
Field[8].FieldsList = {2, 3, 5, 6};


Background Field = 8;
