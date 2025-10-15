SetFactory("OpenCASCADE");

// Setting variables
R_1 = 2; // radius of the first sphere
R_2 = 4; // radius of the second sphere
d = 20; // distance between the centres of the spheres
R_Omega = 48; // radius of the internal domain
Ngamma = 301; // number of nodes on the domain boundary
minSize = 0.05; // size of the elements inside the spheres
maxSize = 0.5; // size of the elements far from the spheres

// First sphere
Point(1) = {0, d/2, 0};
Point(2) = {0, (d/2)+R_1, 0};
Point(3) = {0, (d/2)-R_1, 0};

Circle(1) = {2, 1, 3};
Line(2) = {2, 1};
Line(3) = {1, 3};

Curve Loop(1) = {1, -3, -2};
Plane Surface(1) = {1};

Physical Surface("Sphere_1", 300) = {1};


// Second sphere
Point(4) = {0, -d/2, 0};
Point(5) = {0, -(d/2)+R_2, 0};
Point(6) = {0, -(d/2)-R_2, 0};

Circle(4) = {5, 4, 6};
Line(5) = {5, 4};
Line(6) = {4, 6};

Curve Loop(2) = {4, -6, -5};
Plane Surface(2) = {2};

Physical Surface("Sphere_2", 301) = {2};


// Domain sphere
Point(7) = {0, 0, 0};
Point(8) = {0, R_Omega, 0};
Point(9) = {0, -R_Omega, 0};

Circle(7) = {8, 7, 9};
Line(8) = {8, 2};
Line(9) = {3, 7};
Line(10) = {7, 5};
Line(11) = {6, 9};

Curve Loop(3) = {7, -11, -4, -10, -9, -1, -8};
Plane Surface(3) = {3};

Physical Surface("Internal_domain", 302) = {3};
Physical Curve("Internal_boundary", 200) = {7};


// Mesh size
Transfinite Curve{7} = Ngamma Using Progression 1;


Field[1] = Distance;
Field[1].CurvesList = {1};

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].DistMin = 0;
Field[2].DistMax = 6 * R_1;
Field[2].SizeMin = minSize;
Field[2].SizeMax = maxSize / 2;
Field[2].StopAtDistMax = 1;

Field[3] = Threshold;
Field[3].InField = 1;
Field[3].DistMin = 6 * R_1;
Field[3].DistMax = R_Omega;
Field[3].SizeMin = maxSize / 2;
Field[3].SizeMax = maxSize;
Field[3].StopAtDistMax = 0;

Field[4] = Distance;
Field[4].CurvesList = {4};

Field[5] = Threshold;
Field[5].InField = 4;
Field[5].DistMin = 0;
Field[5].DistMax = 6 * R_2;
Field[5].SizeMin = minSize;
Field[5].SizeMax = maxSize / 2;
Field[5].StopAtDistMax = 1;

Field[6] = Threshold;
Field[6].InField = 4;
Field[6].DistMin = 6 * R_2;
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
