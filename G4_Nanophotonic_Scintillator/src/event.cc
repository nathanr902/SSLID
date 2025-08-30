#include "event.hh"

NSEventAction::NSEventAction(NSRunAction*)
{
    fEdep = 0.;
}

NSEventAction::~NSEventAction()
{}

void NSEventAction::BeginOfEventAction(const G4Event*)
{
    fEdep = 0.;
}

void NSEventAction::EndOfEventAction(const G4Event*)
{
    // #ifndef G4MULTITHREADED
    //     G4cout << "Energy deposition: " << fEdep << G4endl;
    // #endif

    G4AnalysisManager *man = G4AnalysisManager::Instance();
    man->FillNtupleDColumn(2, 0, fEdep);
    man->AddNtupleRow(2);
}