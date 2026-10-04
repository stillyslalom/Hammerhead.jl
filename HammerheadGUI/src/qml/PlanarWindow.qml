// Planar PIV workflow window: the shared window chrome (WorkflowWindow.qml)
// with the planar step pages. Models: `stepModel`, `passModel`, `prepModel`.
import QtQuick
import "steps"

WorkflowWindow {
    stepKeys: ["images", "prepare", "passes", "test", "run", "results"]

    ImagesPage {}
    PreparePage {}
    PassesPage {}
    TestPage {}
    RunPage {}
    ResultsPage {}
}
