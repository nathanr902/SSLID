#ifndef RUN_HH
#define RUN_HH

#include "G4UserRunAction.hh"
#include "G4Run.hh"

#include "G4AnalysisManager.hh"
#include <G4String.hh>
#include "G4GenericMessenger.hh"

class NSRunAction : public G4UserRunAction
{
public:
    NSRunAction();
    ~NSRunAction();
    
    virtual void BeginOfRunAction(const G4Run*);
    virtual void EndOfRunAction(const G4Run*);

private:
    G4GenericMessenger *fMessenger;
    G4String root_file_name;
};

#endif