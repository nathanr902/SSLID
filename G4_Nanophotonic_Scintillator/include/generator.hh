#ifndef GENERATOR_HH
#define GENERATOR_HH

#include "G4VUserPrimaryGeneratorAction.hh"

#include "G4ParticleGun.hh"
#include "G4SystemOfUnits.hh"
#include "G4ParticleTable.hh"
#include "G4IonTable.hh"
#include "G4Gamma.hh"
#include "G4ChargedGeantino.hh"
#include "Randomize.hh"

class NSPrimaryGenerator : public G4VUserPrimaryGeneratorAction
{
public:
    NSPrimaryGenerator();
    ~NSPrimaryGenerator();

    virtual void GeneratePrimaries(G4Event*);

private:
    G4ParticleGun *fParticleGun;
};

#endif