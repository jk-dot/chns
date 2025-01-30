SetFactory("OpenCASCADE");

// Parameters for the torus
MajorRadius = 2;  // Major radius
MinorRadius = 1;  // Minor radius
lc = 0.5;  // Mesh size

// Create a full torus centered at the origin
Torus(1) = {0, 0, 0, MajorRadius, MinorRadius};

// Define the mesh size
Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;

// Generate the mesh
Mesh 3;
Save "torus.msh";
