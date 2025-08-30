#include "construction.hh"
NSDetectorConstruction::NSDetectorConstruction()
{
    // World dimensions
    xWorld = 2000.*um;
    yWorld = 2000.*um;
    zWorld = 2000.*um;

    // The overlaps between the detectors should be checked each time the geometry is changed
    // This is set to false by default to avoid performing this verification at each run
    checkDetectorsOverlaps = false;
    isPbStoppinglayer=false;
    // Construct an array sensitive detectors
    constructDetectors = false;
    nGridX = 1;
    nGridY = 1;
    detectorDepth = 0.1*um;

    // Construct a multilayer structure - The subtrate is considered as a layer
    //isMultilayerNS = false;
    isMultilayerNS = true;
    nLayersNS = 1;
    substrateThickness = 0.5*um;
    //scintillatorThickness = {680.*nm};
    scintillatorThickness = 1.5*um;
    dielectricThickness = 0.2*um;
    //scintillatorThicknessList="1.5";
    //dielectricThicknessList="1.5";
    scintillator_type=1;//default bulk
    startWithScintillator = true;

    // Construct a Au layer embedded into the scintillator
    isAuLayer = false;
    //isAuLayer = true;
    AuLayerThickness = 50.*nm;
    AuLayerDepth = 400.*nm;
    AuLayerSizeX = 1.*um;
    AuLayerSizeY = 1.*um;

    // Scintillation parameters
    scintillatorLifeTime = 2.5*ns;
    scintillationYield = 9000./MeV;

    // Messenger
    fMessenger = new G4GenericMessenger(this, "/structure/", "Structure construction");
    fMessenger->DeclareProperty("xWorld", xWorld, "Size on the x-axis");
    fMessenger->DeclareProperty("yWorld", yWorld, "Size on the y-axis");
    fMessenger->DeclareProperty("zWorld", zWorld, "Size on the z-axis");
    fMessenger->DeclareProperty("nGridX", nGridX, "Grid size in the X axis");
    fMessenger->DeclareProperty("nGridY", nGridY, "Grid size in the Y axis");
    fMessenger->DeclareProperty("isMultilayerNS", isMultilayerNS, "Construct multilayer sintillator");
    fMessenger->DeclareProperty("startWithScintillator", startWithScintillator, "First layer of the multilayer is a sintillator");
    fMessenger->DeclareProperty("constructDetectors", constructDetectors, "Construct detector plane at the end of the structure and not in the materials themselves");
    fMessenger->DeclareProperty("detectorDepth", detectorDepth, "Depth of the detectors");
    fMessenger->DeclareProperty("nLayersNS", nLayersNS, "Number of layers in the nanophotonic scintillator");
    fMessenger->DeclareProperty("scintillatorType", scintillator_type, "type of scintillator to construct");
    fMessenger->DeclareProperty("substrateThickness", substrateThickness, "The thickness of the substrate");
    fMessenger->DeclareProperty("scintillatorThickness", scintillatorThickness, "The thickness of the scintillator layer");
    fMessenger->DeclareProperty("scintillatorThicknessList", scintillatorThicknessList, "The thickness of the scintillator layers");
    //fmessenger->DeclareMethod("scintillatorThickness", &NSDetectorConstruction::SetNewValue, "Set input list for simulation (comma-separated numbers)");
    fMessenger->DeclareProperty("dielectricThickness", dielectricThickness, "The thickness of the dielectric layer");
    fMessenger->DeclareProperty("dielectricThicknessList", dielectricThicknessList, "The thickness of the dielectric layers");
    fMessenger->DeclareProperty("scintillatorLifeTime", scintillatorLifeTime, "The scintillator lifetime");
    fMessenger->DeclareProperty("scintillationYield", scintillationYield, "Scintillation yield");
    fMessenger->DeclareProperty("checkDetectorsOverlaps", checkDetectorsOverlaps, "Check the detectors overlaps");
    fMessenger->DeclareProperty("isAuLayer", isAuLayer, "Construct a metal layer embedded into the scintillator");
    fMessenger->DeclareProperty("AuLayerThickness", AuLayerThickness, "Au Layer Thickness - Tal experiment");
    fMessenger->DeclareProperty("AuLayerDepth", AuLayerDepth, "Au Layer Depth - Tal experiment");
    fMessenger->DeclareProperty("AuLayerSizeX", AuLayerSizeX, "Au Layer size X - Tal experiment");
    fMessenger->DeclareProperty("AuLayerSizeY", AuLayerSizeY, "Au Layer size Y - Tal experiment");
    fMessenger->DeclareProperty("isPbStoppinglayer", isPbStoppinglayer, "determine the type of stopping layer in hybrid scintilator");
}   

NSDetectorConstruction::~NSDetectorConstruction()
{
    delete navigator;
}

G4double NSDetectorConstruction::GetTotalThickness()
{
    if (isMultilayerNS)
    {
        G4double totalThickness = 0.;
        for (G4int l = 0; l < nLayersNS; ++l)
            totalThickness += layerThicknesses[l];
        return totalThickness;
    }
    if (isAuLayer)
    {
        return substrateThickness + scintillatorThickness;
    }
    return 0.;
}

float wavelenthToeV(float wavelength)
{
    return 1239.84193 / (wavelength / nm) * eV;
}

void DefineScintillator(G4MaterialPropertiesTable* mpt, G4Material* material, const G4int numComponents, G4double* energy, G4double* rindex, G4double* fraction, G4double* absLength, const float yield, const float resolutionScale, const float timeConstant)
{
    mpt->AddProperty("RINDEX", energy, rindex, numComponents);
    // mpt->AddProperty("ABSLENGTH", energy, absLength, numComponents);
    mpt->AddConstProperty("SCINTILLATIONYIELD", yield);
    mpt->AddConstProperty("RESOLUTIONSCALE",resolutionScale);
    mpt->AddProperty("SCINTILLATIONCOMPONENT1", energy, fraction, numComponents);
    mpt->AddConstProperty("SCINTILLATIONYIELD1", 1.);
    mpt->AddConstProperty("SCINTILLATIONTIMECONSTANT1", timeConstant);
    material->SetMaterialPropertiesTable(mpt);
    material->GetIonisation()->SetBirksConstant(0.126 * mm / MeV);
}

G4MaterialPropertiesTable* DefineNonScintillatingMaterial(G4Material* material, G4int nComponents, G4double* energy, G4double* rindex, G4double* absLength)
{
    G4MaterialPropertiesTable *mpt = new G4MaterialPropertiesTable();
    mpt->AddProperty("RINDEX", energy, rindex, nComponents);
    // mpt->AddProperty("ABSLENGTH", energy, absLength, nComponents);
    material->SetMaterialPropertiesTable(mpt);
    return mpt;
}

void NSDetectorConstruction::DefineOpticalSurface(G4MaterialPropertiesTable* mpt, G4OpticalSurface* opticalSurface, G4String opticalSurfaceName)
{
    opticalSurface->SetType(dielectric_dielectric);
    opticalSurface->SetFinish(polished);
    opticalSurface->SetMaterialPropertiesTable(mpt);
}

void NSDetectorConstruction::DefineElements(G4NistManager *nist)
{
    C = nist->FindOrBuildElement("C");
    Ti = nist->FindOrBuildElement("Ti");
    H = nist->FindOrBuildElement("H");
    O = nist->FindOrBuildElement("O");
    Na = nist->FindOrBuildElement("Na");
    I = nist->FindOrBuildElement("I");
    Si = nist->FindOrBuildElement("Si");
    Cl = nist->FindOrBuildElement("Cl");
    Au = nist->FindOrBuildElement("Au");
    Pb = nist->FindOrBuildElement("Pb");
}

void NSDetectorConstruction::DefineAuLayer(G4double* energy)
{
    AuLayer = new G4Material("AuLayer", 19.3*g/cm3, 1);
    AuLayer->AddElement(Au, 1);

    G4MaterialPropertiesTable *mptAu = new G4MaterialPropertiesTable();
    AuLayer->SetMaterialPropertiesTable(mptAu);
    
    // Creating the optical surface properties
    opticalSurfaceAu = new G4OpticalSurface("interfaceSurfaceAu");
    DefineOpticalSurface(mptAu, opticalSurfaceAu, "interfaceSurfaceAu");
}

void NSDetectorConstruction::DefineWorld(G4NistManager *nist, G4double* energy)
{
    worldMat = nist->FindOrBuildMaterial("G4_AIR");
    G4double rindexWorld[2] = {1.0, 1.0};
    G4double fraction[2] = {1.0, 1.0};
    G4MaterialPropertiesTable *mptWorld = new G4MaterialPropertiesTable();
    mptWorld->AddProperty("RINDEX", energy, rindexWorld, 2);
    worldMat->SetMaterialPropertiesTable(mptWorld);

    // // Creating the optical surface properties
    opticalSurfaceWorld = new G4OpticalSurface("interfaceSurfaceWorld");
    DefineOpticalSurface(mptWorld, opticalSurfaceWorld, "interfaceSurfaceWorld");  
}

void NSDetectorConstruction::DefineNaI(G4NistManager *nist, G4double* energy)
{
    NaI = new G4Material("NaI", 3.67*g/cm3, 2);
    NaI->AddElement(Na, 1);
    NaI->AddElement(I, 1);

    G4double fraction[2] = {1.0, 1.0};
    G4double rindexNaI[2] = {1.78, 1.78};
    G4double absLengthNaI[2] = {0.3*mm, 0.3*mm};
    G4MaterialPropertiesTable* mptNaI = new G4MaterialPropertiesTable();
    DefineScintillator(mptNaI, NaI, 2, energy, rindexNaI, fraction, absLengthNaI, 38./keV, 1., 10.*ns);

    // // Creating the optical surface properties
    opticalSurfaceNaI = new G4OpticalSurface("interfaceSurfaceNaI");
    DefineOpticalSurface(mptNaI, opticalSurfaceNaI, "interfaceSurfaceNaI");
}

void NSDetectorConstruction::DefineSiO2(G4double* energy)
{
    SiO2 = new G4Material("SiO2", 2.201*g/cm3, 2);
    SiO2->AddElement(Si, 1);
    SiO2->AddElement(O, 2);

    G4double rindexSiO2[2] = {1.45, 1.45};
    G4double absLengthSiO2[2] = {25.*um, 25.*um};
    G4MaterialPropertiesTable *mptSiO2 = DefineNonScintillatingMaterial(SiO2, 2, energy, rindexSiO2, absLengthSiO2);

    // Creating the optical surface properties
    opticalSurfaceSiO2 = new G4OpticalSurface("interfaceSurfaceSiO2");
    DefineOpticalSurface(mptSiO2, opticalSurfaceSiO2, "interfaceSurfaceSiO2");
}

void NSDetectorConstruction::DefinePbStopping(G4double* energy)
{
    Pb_stopping = new G4Material("Pb", 11.34*g/cm3, 1);
    Pb_stopping->AddElement(Pb, 1);

    G4double rindexPb_stopping[2] = {1.9, 1.9};
    G4double absLengthPb_stopping[2] = {25.*um, 25.*um};
    G4MaterialPropertiesTable *mptPb = DefineNonScintillatingMaterial(Pb_stopping, 2, energy, rindexPb_stopping, absLengthPb_stopping);

    // Creating the optical surface properties
    opticalSurfacePb = new G4OpticalSurface("interfaceSurfacePb");

    DefineOpticalSurface(mptPb, opticalSurfacePb, "opticalSurfacePb");
}
void NSDetectorConstruction::DefinePVT(G4double* energy)
{
    PVT = new G4Material("PVT", 1.02*g/cm3, 2);
    PVT->AddElement(C, 9);
    PVT->AddElement(H, 10);
    
    G4double rindexPVT[2] = {1.61, 1.61};
    G4double absLengthPVT[2] = {300.*um, 300.*um};
    G4double fraction[2] = {1.0, 1.0};
    G4MaterialPropertiesTable* mptPVT = new G4MaterialPropertiesTable();
    DefineScintillator(mptPVT, PVT, 2, energy, rindexPVT, fraction, absLengthPVT, scintillationYield, 1., scintillatorLifeTime);

    // Creating the optical surface properties
    opticalSurfacePVT = new G4OpticalSurface("interfaceSurfacePVT");
    DefineOpticalSurface(mptPVT, opticalSurfacePVT, "interfaceSurfacePVT");
}

void NSDetectorConstruction::DefineStoppingLayer(G4double* energy)
{
    stoppingMaterial = new G4Material("stoppingMaterial", 2.3*g/cm3, 4);
    stoppingMaterial->AddElement(Ti, 0.4);
    stoppingMaterial->AddElement(O, 0.4);
    stoppingMaterial->AddElement(Cl, 0.07);
    stoppingMaterial->AddElement(C, 0.13);

    G4double rindexSL[2] = {1.9, 1.9};
    G4double absLengthSL[2] = {25.*um, 25.*um};
    G4MaterialPropertiesTable *mptSL = DefineNonScintillatingMaterial(stoppingMaterial, 2, energy, rindexSL, absLengthSL);
    
    // Creating the optical surface properties
    opticalSurfaceStoppingLayer = new G4OpticalSurface("opticalSurfaceStoppingLayer");
    DefineOpticalSurface(mptSL, opticalSurfaceStoppingLayer, "opticalSurfaceStoppingLayer");
}
void NSDetectorConstruction::DefineLeadGlassStoppingLayer(G4double* energy)
{
    stoppingMaterial = new G4Material("stoppingMaterial", 5*g/cm3, 3);
    stoppingMaterial->AddElement(Pb, 0.2);
    stoppingMaterial->AddElement(O, 0.5);
    stoppingMaterial->AddElement(Si, 0.3);
    //stoppingMaterial->AddElement(O, 0.13);

    G4double rindexSL[2] = {1.9, 1.9};
    G4double absLengthSL[2] = {25.*um, 25.*um};
    G4MaterialPropertiesTable *mptSL = DefineNonScintillatingMaterial(stoppingMaterial, 2, energy, rindexSL, absLengthSL);
    
    // Creating the optical surface properties
    opticalSurfaceStoppingLayer = new G4OpticalSurface("opticalSurfaceStoppingLayer");
    DefineOpticalSurface(mptSL, opticalSurfaceStoppingLayer, "opticalSurfaceStoppingLayer");
}

void NSDetectorConstruction::DefineH2O(G4double* energy)
{
    H2O = new G4Material("H2O", 1.000*g/cm3, 2);
    H2O->AddElement(H, 2);
    H2O->AddElement(O, 1);

    G4double rindexH2O[2] = {1., 1.};
    G4double absLengthH2O[2] = {7.6*mm, 7.6*mm};
    G4MaterialPropertiesTable *mptH2O = DefineNonScintillatingMaterial(H2O, 2, energy, rindexH2O, absLengthH2O);

    // Creating the optical surface properties
    opticalSurfaceH2O = new G4OpticalSurface("interfaceSurfaceH2O");
    DefineOpticalSurface(mptH2O, opticalSurfaceH2O, "interfaceSurfaceH2O");
}

void NSDetectorConstruction::DefineMaterials()
{
    G4NistManager *nist = G4NistManager::Instance();

    G4double energy[2] = {wavelenthToeV(440*nm), wavelenthToeV(430*nm)};
    DefineElements(nist);
    DefineWorld(nist, energy);
    DefineNaI(nist, energy);
    DefineSiO2(energy);
    DefinePVT(energy);
    //DefineStoppingLayer(energy);
    //DefinePbStopping(energy);
    DefineLeadGlassStoppingLayer(energy);
    DefineH2O(energy);
    DefineAuLayer(energy);
}
void NSDetectorConstruction::ConstructBulkNS()
{
// Constructing the substrate
    //solidMultilayerNS = new G4Box("solidSubstrate", xWorld, yWorld, 0.5*substrateThickness);
    //logicMultilayerNS = new G4LogicalVolume(solidMultilayerNS, SiO2, "logicSubstrate");
    //G4VPhysicalVolume *physVol= new G4PVPlacement(0, G4ThreeVector(0., 0., 0.5*substrateThickness), logicMultilayerNS, "physSubstrate", logicWorld, false, 0, checkDetectorsOverlaps);

    // Construct the scintillator
    G4Box *solidScintillator = new G4Box("solidScintillator", xWorld, yWorld, 0.5*scintillatorThickness);
    G4LogicalVolume *logicScintillator = new G4LogicalVolume(solidScintillator, PVT, "logicScintillator");
    
    G4VPhysicalVolume *physScintillator = new G4PVPlacement(0, G4ThreeVector(0., 0., 0.5*scintillatorThickness), logicScintillator, "physScintillator", logicWorld, false, 0, checkDetectorsOverlaps);
    
    new G4LogicalBorderSurface("interfaceWorldScintillator", physWorld, physScintillator, opticalSurfaceWorld);
    new G4LogicalBorderSurface("interfaceScintillatorWorld", physScintillator, physWorld, opticalSurfaceScintillator);
    //new G4LogicalBorderSurface("interfaceScintillatorSubstrate", physScintillator, physVol, opticalSurfaceScintillator);
    //new G4LogicalBorderSurface("interfaceSubstrateScintillator", physVol, physScintillator, opticalSurfaceSiO2);

    const G4Colour scintillatorColor(0.,0.,0.7);
    const G4VisAttributes *scintillatorVisAttributes = new G4VisAttributes(scintillatorColor);
    logicScintillator->SetVisAttributes(scintillatorVisAttributes);
}
std::vector<G4double> NSDetectorConstruction::generate_thickness(const std::string& input) {
    std::vector<G4double> result;
    std::istringstream ss(input);
    std::string token;
    
    while (std::getline(ss, token, ',')) {
        G4double number;
        std::istringstream(token) >> number;
        result.push_back(number);
    }
    
    return result;
}

void NSDetectorConstruction::ConstructMultilayerNS()
{
    if (scintillator_type == 0 )// bulk
    {
        // Adding a substrate
        //layerThicknesses.push_back(substrateThickness);
        //layerIsScintillator.push_back(false);
        //layerMaterials.push_back(SiO2);
        // Place the layers on top of the substrate
        layerThicknesses.push_back(scintillatorThickness);
        layerIsScintillator.push_back(true);
        layerMaterials.push_back(PVT);
    }
    else if (scintillator_type == 1) // periodic
    {
        // Adding a substrate
        //layerThicknesses.push_back(substrateThickness);
        //layerIsScintillator.push_back(false);
        //layerMaterials.push_back(SiO2);
        // Place the layers on top of the substrate
        for (G4int l = 0; l < nLayersNS; ++l)
        {
            if (l % 2 != startWithScintillator)
            {
                layerThicknesses.push_back(scintillatorThickness);
                layerIsScintillator.push_back(true);
                layerMaterials.push_back(PVT);
            } else {
                layerThicknesses.push_back(dielectricThickness);
                layerIsScintillator.push_back(false);
                layerMaterials.push_back(stopping_material_g);
            }
        }
        
    }
    else if (scintillator_type == 2) // aperiodic
    {
        scintLayerThicknesses=generate_thickness(scintillatorThicknessList);
        dielectricLayerThicknesses=generate_thickness(dielectricThicknessList);
        for (G4int l = 0; l < nLayersNS; ++l)
        {
            if (l % 2 != startWithScintillator)
            {
                l_thick=scintLayerThicknesses[0];
                scintLayerThicknesses.erase(scintLayerThicknesses.begin());
                layerThicknesses.push_back(l_thick);
                layerIsScintillator.push_back(true);
                layerMaterials.push_back(PVT);
            } else { 
                l_thick=dielectricLayerThicknesses[0];
                dielectricLayerThicknesses.erase(dielectricLayerThicknesses.begin());
                layerThicknesses.push_back(l_thick);
                layerIsScintillator.push_back(false);
                layerMaterials.push_back(stopping_material_g);
            }
        }
        
    }
    G4LogicalBorderSurface *interfaceSurface;
    // Constructing the multilayer structure
    G4double cumDepthOffset = 0.;
    for(G4int l = 0; l < nLayersNS; ++l)
    {   
        solidMultilayerNS = new G4Box("solidMultilayerNS"+std::to_string(l), xWorld, yWorld, 0.5*layerThicknesses[l]);
        logicMultilayerNS = new G4LogicalVolume(solidMultilayerNS, layerMaterials[l], "logicMultilayerNS"+std::to_string(l));
        if (layerIsScintillator[l])
        {
            const G4Colour scintillatorColor(0.8,0.,0.3);
            const G4VisAttributes *scintillatorVisAttributes = new G4VisAttributes(scintillatorColor);
            logicMultilayerNS->SetVisAttributes(scintillatorVisAttributes);
        }
        if (!constructDetectors){
            NSSensitiveDetector *sensDet = new NSSensitiveDetector("SensitiveDetector"+std::to_string(l), constructDetectors);
            logicMultilayerNS->SetSensitiveDetector(sensDet);
            fScoringVolumes.push_back(logicMultilayerNS);
        }

        cumDepthOffset += 0.5*layerThicknesses[l];
        G4ThreeVector trans(0., 0., cumDepthOffset);
        
        physMultilayerNSArray.push_back(new G4PVPlacement(0, trans, logicMultilayerNS, "physMultilayerNS", logicWorld, false, l, checkDetectorsOverlaps));
        cumDepthOffset += 0.5*layerThicknesses[l];

        // Assign optical surface to the interface
        if (l == 0) //setting interface between the world and the first layer
        {
            interfaceSurface = new G4LogicalBorderSurface("interface"+std::to_string(l), physWorld, physMultilayerNSArray.back(), opticalSurfaceWorld);
        } else  // Between the first and last layers
        {
            const auto& opticalSurface = layerIsScintillator[l - 1] ? opticalSurfaceScintillator : opticalSurfaceDielectric;
            interfaceSurface = new G4LogicalBorderSurface("interface"+std::to_string(l), physMultilayerNSArray.end()[-2], physMultilayerNSArray.back(), opticalSurface);
        } 
        if (l == (nLayersNS - 1)) // Last layer
        {
            const auto& lastOpticalSurface = layerIsScintillator[l] ? opticalSurfaceScintillator : opticalSurfaceDielectric;
            interfaceSurface = new G4LogicalBorderSurface("interface"+std::to_string(l+1), physMultilayerNSArray.back(), physWorld, lastOpticalSurface);
        }
    }
}

void NSDetectorConstruction::ConstructSensitiveDetector()
{
    // Constructing the cells of the detector
    const G4double detectorWidth = 2.*xWorld/nGridX, detectorHeight = 2.*yWorld/nGridY;
    solidDetector = new G4Box("solidDetector", 0.5*detectorWidth, 0.5*detectorHeight, 0.5*detectorDepth);
    logicDetector = new G4LogicalVolume(solidDetector, worldMat, "logicDetector");
    logicDetector->SetUserLimits(userLimits);
    NSSensitiveDetector *sensDet = new NSSensitiveDetector("SensitiveDetector", constructDetectors);
    logicDetector->SetSensitiveDetector(sensDet);
    for(G4int i = 0; i < nGridX; i++)
    {
        for(G4int j = 0; j < nGridY; j++)
        {
            physDetector = new G4PVPlacement(0, G4ThreeVector(-xWorld+(i+0.5)*detectorWidth, -yWorld+(j+0.5)*detectorHeight, GetTotalThickness()+0.5*detectorDepth), logicDetector, "physDetector", logicWorld, false, j+i*nGridY, checkDetectorsOverlaps);
        }
    }
}

void NSDetectorConstruction::ConstructAuLayer()
{
    // Constructing the substrate
    solidMultilayerNS = new G4Box("solidSubstrate", xWorld, yWorld, 0.5*substrateThickness);
    logicMultilayerNS = new G4LogicalVolume(solidMultilayerNS, SiO2, "logicSubstrate");
    G4VPhysicalVolume *physVol= new G4PVPlacement(0, G4ThreeVector(0., 0., 0.5*substrateThickness), logicMultilayerNS, "physSubstrate", logicWorld, false, 0, checkDetectorsOverlaps);

    // Construct the scintillator
    G4Box *solidScintillator = new G4Box("solidScintillator", xWorld, yWorld, 0.5*scintillatorThickness);
    G4LogicalVolume *logicScintillator = new G4LogicalVolume(solidScintillator, PVT, "logicScintillator");
    
    G4VPhysicalVolume *physScintillator = new G4PVPlacement(0, G4ThreeVector(0., 0., 0.5*scintillatorThickness + substrateThickness), logicScintillator, "physScintillator", logicWorld, false, 0, checkDetectorsOverlaps);
    
    new G4LogicalBorderSurface("interfaceWorldScintillator", physWorld, physScintillator, opticalSurfaceWorld);
    new G4LogicalBorderSurface("interfaceScintillatorWorld", physScintillator, physWorld, opticalSurfaceScintillator);
    new G4LogicalBorderSurface("interfaceScintillatorSubstrate", physScintillator, physVol, opticalSurfaceScintillator);
    new G4LogicalBorderSurface("interfaceSubstrateScintillator", physVol, physScintillator, opticalSurfaceSiO2);

    const G4Colour scintillatorColor(0.,0.,0.7);
    const G4VisAttributes *scintillatorVisAttributes = new G4VisAttributes(scintillatorColor);
    logicScintillator->SetVisAttributes(scintillatorVisAttributes);

    // Constructing the Au layer
    G4Box *solidAu = new G4Box("solidAuLayer", 0.5*AuLayerSizeX,0.5*AuLayerSizeY, 0.5*AuLayerThickness);
    G4LogicalVolume *logicAu = new G4LogicalVolume(solidAu, AuLayer, "logicAuLayer");
    
    G4VPhysicalVolume *physAu = new G4PVPlacement(0, G4ThreeVector(0., 0., -0.5*scintillatorThickness + AuLayerDepth), logicAu, "physMultilayerNSAu", logicScintillator, false, 1, checkDetectorsOverlaps);
    
    new G4LogicalBorderSurface("interfaceSilicaAu", physAu, physScintillator, opticalSurfaceSiO2);

    const G4Colour AuColor(0.83,0.69,0.21);
    const G4VisAttributes *AuVisAttributes = new G4VisAttributes(AuColor);
    logicAu->SetVisAttributes(AuVisAttributes);
}

G4VPhysicalVolume *NSDetectorConstruction::Construct()
{
    DefineMaterials();
    
    navigator = new G4Navigator();
    solidWorld = new G4Box("solidWorld", xWorld, yWorld, zWorld);
    logicWorld = new G4LogicalVolume(solidWorld, worldMat, "logicWorld");
    physWorld = new G4PVPlacement(0, G4ThreeVector(0., 0., 0.), logicWorld, "physWorld", 0, false, 0, checkDetectorsOverlaps);
    navigator->SetWorldVolume(physWorld);

    layerThicknesses.clear(); layerIsScintillator.clear(); layerMaterials.clear();
    opticalSurfaceScintillator = opticalSurfacePVT;
    if(!isPbStoppinglayer)
    {
        opticalSurfaceDielectric = opticalSurfaceStoppingLayer;
        stopping_material_g=stoppingMaterial;
    }
    else
    {
        opticalSurfaceDielectric = opticalSurfacePb;
        stopping_material_g=Pb_stopping;
    }
    
    // userLimits = new G4UserLimits();
    // userLimits->SetMaxAllowedStep(fStepLimit);

    if(isMultilayerNS)
    {       

        ConstructMultilayerNS();

    } else if (isAuLayer) {
        ConstructAuLayer();
    }
    else if(constructDetectors)
    {
        ConstructSensitiveDetector();
    }

    return physWorld;
}

void NSDetectorConstruction::ConstructSDandField() {}