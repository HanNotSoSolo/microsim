SetFactory("OpenCASCADE");

// Setting variables
R_Omega = 12.002; // radius of the internal domain
Ngamma = 75; // number of nodes on the domain boundary
minSize = 0.05; // size of the elements inside the spheres
maxSize = 0.5; // size of the elements far from the spheres

// Domain sphere
Point(1) = {0, 0, 0};
Point(2) = {0, R_Omega, 0};
Point(3) = {0, -R_Omega, 0};

Circle(1) = {2, 1, 3};
Line(2) = {2, 1};
Line(3) = {1, 3};

Curve Loop(1) = {1, -3, -2};
Plane Surface(1) = {1};

Point{1} In Surface{1};

Physical Surface("external_domain", 303) = {1};
Physical Curve("External_boundary", 201) = {1};
Physical Point(0) = {1};


// Mesh size
Transfinite Curve{1} = Ngamma Using Progression 1;

Field[1] = Distance;
Field[1].PointsList = {1};

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].DistMin = 0;
Field[2].DistMax = R_Omega;
Field[2].SizeMin = minSize;
Field[2].SizeMax = maxSize;


Background Field = 2;
