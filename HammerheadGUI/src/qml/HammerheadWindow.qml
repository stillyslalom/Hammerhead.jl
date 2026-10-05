// The Hammerhead window: the shared window chrome (WorkflowWindow.qml) with
// every step page. The session's recording type (`app.modality`) picks the
// Images page variant and the step rail (Calibration only for stereo).
// Models: `stepModel`, `passModel`, `prepModel`, and the Calibration step's
// plate lists `plates1Model`/`plates2Model`.
import QtQuick
import "steps"

WorkflowWindow {
    stepKeys: ["images", "calibration", "prepare", "passes", "run", "results"]
    stereo: app.modality === "stereo"

    Item {
        ImagesPage { anchors.fill: parent; visible: app.modality !== "stereo" }
        StereoImagesPage { anchors.fill: parent; visible: app.modality === "stereo" }
    }
    CalibrationPage {}
    PreparePage { stereo: app.modality === "stereo" }
    PassesPage { particleModes: app.modality !== "stereo" }
    RunPage {}
    ResultsPage {}
}
