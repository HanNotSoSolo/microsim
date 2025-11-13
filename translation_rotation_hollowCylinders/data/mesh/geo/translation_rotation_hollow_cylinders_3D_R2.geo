SetFactory("OpenCASCADE");

// Setting variables
// First cylinder (the one that can move)
R_int_1 = 0.0154; // internal radius of the first cylinder
R_ext_1 = 0.0197; // external radius of the first cylinder
h_1 = 0.04337; // height of the first cylinder
Z_1 = -1e-03; // vertical displacement of the first cylinder
R_1 = 5e-3; // radial displacement of the first cylinder
R_int_2 = 0.0304; // internal radius of the second cylinder
R_ext_2 = 0.0346975; // external radius of the second cylinder
h_2 = 0.07983; // height of the second cylinder
R_Omega = 0.10577568210604932; // radius of the internal domain
Ngamma = 474; // number of nodes on the domain boundary
minSize = 0.0005; // size of the elements inside the spheres
maxSize = 0.005; // size of the elements far from the spheres
d = 4.001; // distance between the centres of the spheres

// First cylinder
Point(1) = {0, R_int_1+R_1, -h_1/2+Z_1};
Point(2) = {0, R_ext_1+R_1, -h_1/2+Z_1};
Point(3) = {0, R_ext_1+R_1, h_1/2+Z_1};
Point(4) = {0, R_int_1+R_1, h_1/2+Z_1};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Extrude { {0, 0, 1}, {0, R_1, Z_1}, 2*Pi} {
    Surface{1}; Layers{7}; Recombine;
    }

Physical Volume("Hollow_cylinder_1", 300) = {1};


// Second cylinder
Point(5) = {0, R_int_2, -h_2/2};
Point(6) = {0, R_ext_2, -h_2/2};
Point(7) = {0, R_ext_2, h_2/2};
Point(8) = {0, R_int_2, h_2/2};

Line(9) = {5, 6};
Line(10) = {6, 7};
Line(11) = {7, 8};
Line(12) = {8, 5};

Curve Loop(8) = {9, 10, 11, 12};
Plane Surface(6) = {8};

Extrude { {0, 0, 1}, {0, 0, 0}, 2*Pi} {
    Surface{6}; Layers{7}; Recombine;
    }

Physical Volume("Hollow_cylinder_2", 301) = {2};


// Mesh size

Field[1] = Constant;
Field[1].VolumesList = {1, 2};
Field[1].VIn = minSize;
Field[1].VOut = maxSize;


Background Field = 1;
