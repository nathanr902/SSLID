#include "run.hh"

NSRunAction::NSRunAction()
{
    G4AnalysisManager *man = G4AnalysisManager::Instance();

    man->CreateNtuple("Photons", "Photons");
    man->CreateNtupleIColumn("fEvent");
    man->CreateNtupleDColumn("fX");
    man->CreateNtupleDColumn("fY");
    man->CreateNtupleDColumn("fZ");
    man->CreateNtupleDColumn("fT");
    man->CreateNtupleDColumn("fWlen");
    man->CreateNtupleSColumn("fType");
    man->CreateNtupleDColumn("fTrackLength");
    man->CreateNtupleSColumn("fMaterial");
    man->CreateNtupleSColumn("fProcess");
    man->CreateNtupleIColumn("fStepStatusNumber");
    man->CreateNtupleIColumn("fProcessType");
    man->CreateNtupleIColumn("fSubProcessType");
    man->CreateNtupleSColumn("fCreatorProcess");
    man->CreateNtupleIColumn("ftrackID");
    man->CreateNtupleIColumn("fParentTrackID");
    man->CreateNtupleIColumn("FirstStep");
    man->CreateNtupleIColumn("StepIndex");
    man->CreateNtupleIColumn("Energy");
    man->FinishNtuple(0);

    man->CreateNtuple("Hits", "Hits");
    man->CreateNtupleIColumn("fEvent");
    man->CreateNtupleDColumn("fX");
    man->CreateNtupleDColumn("fY");
    man->CreateNtupleDColumn("fZ");
    man->FinishNtuple(1);

    man->CreateNtuple("Scoring", "Scoring");
    man->CreateNtupleDColumn("fEdep");
    man->FinishNtuple(2);

    root_file_name = "output.root";
    fMessenger = new G4GenericMessenger(this, "/system/", "System params");
    fMessenger->DeclareProperty("root_file_name", root_file_name, "Grid size in the X axis");

}

NSRunAction::~NSRunAction()
{}

void NSRunAction::BeginOfRunAction(const G4Run* run)
{
    G4AnalysisManager *man = G4AnalysisManager::Instance();
    if (root_file_name == "output.root")
    {
        G4int runID = run->GetRunID();
        std::stringstream strRunID;
        strRunID << runID;
        root_file_name = "output"+strRunID.str()+".root";
    }
    man->OpenFile(root_file_name);
}

void NSRunAction::EndOfRunAction(const G4Run*)
{
    G4AnalysisManager *man = G4AnalysisManager::Instance();
    man->Write();
    man->CloseFile();
}