#ifndef CONSTRUCTION_HH
#define CONSTRUCTION_HH

#include "G4VUserDetectorConstruction.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4PVPlacement.hh"
#include "G4NistManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4GenericMessenger.hh"
#include "G4OpticalSurface.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4Colour.hh"
#include "G4VisAttributes.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4UserLimits.hh"

#include "detector.hh"
#include <G4Types.hh>

class NSDetectorConstruction : public G4VUserDetectorConstruction
{
public:
    NSDetectorConstruction();
    ~NSDetectorConstruction();

    std::vector<G4LogicalVolume*> GetScoringVolume() const { return fScoringVolumes; }

    virtual G4VPhysicalVolume *Construct();
    void ConstructMultilayerNS();
    void ConstructSensitiveDetector();
    G4double GetTotalThickness();
    void ConstructAuLayer();
    void ConstructBulkNS();
private:
    G4Navigator* navigator;

    G4Box *solidWorld, *solidDetector;
    G4LogicalVolume *logicWorld, *logicDetector;
    G4VPhysicalVolume *physWorld, *physDetector;

    G4Box* solidMultilayerNS;
    G4LogicalVolume* logicMultilayerNS;
    std::vector<G4VPhysicalVolume*> physMultilayerNSArray;
    G4OpticalSurface *mirrorSurface;

    G4UserLimits* userLimits;
    float fStepLimit;
    G4Material *stopping_material_g;
    G4Material *SiO2, *H2O, *worldMat, *NaI, *PVA, *TiOH, *stoppingMaterial, *PVT,*Pb_stopping;
    G4Element *C, *Na, *I, *Ti, *H, *O, *Si, *Cl,*Pb;

    bool isAuLayer;
    G4Element *Au;
    G4Material *AuLayer;
    G4OpticalSurface *opticalSurfaceAu;
    G4double AuLayerDepth, AuLayerThickness, AuLayerSizeX, AuLayerSizeY;

    G4OpticalSurface *opticalSurfaceWorld, *opticalSurfaceNaI, *opticalSurfaceSiO2, *opticalSurfacePVT, *opticalSurfaceStoppingLayer,*opticalSurfacePb, *opticalSurfaceH2O;
    G4OpticalSurface *opticalSurfaceScintillator, *opticalSurfaceDielectric;
    std::vector<G4double> generate_thickness(const std::string& input);
    void DefineMaterials();
    void DefineElements(G4NistManager *nist);
    void DefineOpticalSurface(G4MaterialPropertiesTable* mpt, G4OpticalSurface* opticalSurface, G4String opticalSurfaceName);

    void DefineWorld(G4NistManager *nist, G4double* energy);
    void DefineNaI(G4NistManager *nist, G4double* energy);
    void DefineSiO2(G4double* energy);
    void DefinePVT(G4double* energy);
    void DefineStoppingLayer(G4double* energy);
    void DefineLeadGlassStoppingLayer(G4double* energy);
    void DefinePbStopping (G4double* energy);

    void DefineH2O(G4double* energy);
    void DefineAuLayer(G4double* energy);

    virtual void ConstructSDandField();

    G4GenericMessenger *fMessenger;
    std::vector<G4LogicalVolume*> fScoringVolumes;
    G4int nGridX, nGridY;
    G4double xWorld, yWorld, zWorld;
    G4double detectorDepth;

    G4bool isMultilayerNS, constructDetectors, startWithScintillator,isPbStoppinglayer;
    G4int nLayersNS;
    G4int scintillator_type;
    //G4double substrateThickness, scintillatorThickness, dielectricThickness;
    G4double substrateThickness,dielectricThickness,scintillatorThickness;
    //std::vector<G4double> dielectricThickness;
    std::string scintillatorThicknessList,dielectricThicknessList;
    std::vector<G4double> layerThicknesses;
    std::vector<G4double> scintLayerThicknesses,dielectricLayerThicknesses;
    std::vector<G4Material*> layerMaterials;
    std::vector<G4bool> layerIsScintillator;
    G4double scintillatorLifeTime;
    G4double scintillationYield,l_thick;

    bool checkDetectorsOverlaps;
};

#endif