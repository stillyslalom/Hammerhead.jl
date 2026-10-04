// Stereo PIV workflow window: the shared window chrome (WorkflowWindow.qml)
// with the stereo Images, Calibration, and Prepare pages and the shared
// Passes, Test pair, Run, and Results pages. Models: `stepModel`,
// `passModel`, `prepModel`, and the plate lists `plates1Model`/`plates2Model`.
import QtQuick
import "steps"

WorkflowWindow {
    stepKeys: ["images", "calibration", "prepare", "passes", "test", "run", "results"]
    stereo: true

    StereoImagesPage {}
    CalibrationPage {}
    PreparePage { stereo: true }
    PassesPage {}
    TestPage {}
    RunPage {}
    ResultsPage {}
}
