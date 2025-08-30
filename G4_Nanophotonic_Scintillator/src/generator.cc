#include "generator.hh"

NSPrimaryGenerator::NSPrimaryGenerator()
{   
    G4int n_particles = 1;
    //G4int n_particles = 100;
    fParticleGun = new G4ParticleGun(n_particles);

    G4ParticleTable *particleTable = G4ParticleTable::GetParticleTable();
    G4ParticleDefinition *particle = particleTable->FindParticle("gamma");

    G4ThreeVector pos(0.,0.,-5.*um);
    G4ThreeVector mom(0.,0.,1.);
    G4double sigmaX = 0.01 * mm; // Adjust this value for the desired width
    G4double sigmaY = 0.01 * mm;
    //fParticleGun->SetParticlePosition(G4ThreeVector(G4RandGauss::shoot(0.0, sigmaX), G4RandGauss::shoot(0.0, sigmaY), -5.*um));
    fParticleGun->SetParticlePosition(pos);
    fParticleGun->SetParticleMomentumDirection(mom);
    fParticleGun->SetParticleEnergy(10.*keV);
    fParticleGun->SetParticleDefinition(particle);
}

NSPrimaryGenerator::~NSPrimaryGenerator()
{
    delete fParticleGun;
}

void NSPrimaryGenerator::GeneratePrimaries(G4Event *anEvent)
{
    fParticleGun->GeneratePrimaryVertex(anEvent);
}