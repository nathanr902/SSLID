#include "stepping.hh"
#include "G4OpticalPhoton.hh" 
NSSteppingAction::NSSteppingAction(NSEventAction *eventAction)
{
    fEventAction = eventAction;
}

NSSteppingAction::~NSSteppingAction()
{}

void NSSteppingAction::UserSteppingAction(const G4Step *step)
{   
    // G4LogicalVolume *volume = step->GetPreStepPoint()->GetTouchableHandle()->GetVolume()->GetLogicalVolume();
    // const NSDetectorConstruction *detectorConstruction = static_cast<const NSDetectorConstruction*> (G4RunManager::GetRunManager()->GetUserDetectorConstruction());
    // std::vector<G4LogicalVolume*> fScoringVolumes = detectorConstruction->GetScoringVolume();
    
    // for (auto& fScoringVolume: fScoringVolumes)
    // {
    //     if(volume != fScoringVolume)
    //         return;
        
    //     G4double edep = step->GetTotalEnergyDeposit();
    //     fEventAction->AddEdep(edep);
    // }
    G4Track* track = step->GetTrack();
    G4ParticleDefinition* particle = track->GetDefinition();
    // Check if the particle is a lepton (electron or positron).
    if (particle == G4Electron::ElectronDefinition()) {
        
        // Kill the particle if its energy is below 100 eV.
        if (track->GetKineticEnergy() <100.0 *eV) {
            const_cast<G4Step*>(step)->SetTotalEnergyDeposit(0);
            track->SetTrackStatus(fStopAndKill);
            const std::vector<const G4Track*>* secondaries = step->GetSecondaryInCurrentStep();

            // Loop through all secondaries and destroy them
            for (const auto& secondary : *secondaries) {
                // Kill the secondary track
                const_cast<G4Track*>(secondary)->SetTrackStatus(fKillTrackAndSecondaries);
            }
        }
    }
    else if (particle == G4OpticalPhoton::OpticalPhotonDefinition())
    {
        track->SetTrackStatus(fStopAndKill);
    }
}