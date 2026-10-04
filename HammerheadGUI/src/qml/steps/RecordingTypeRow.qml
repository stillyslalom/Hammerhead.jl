// The session's recording type, at the top of the Images step: one camera
// (planar PIV and particle analysis) or two (stereo PIV). Changing it starts
// a fresh session; the window asks first when that would discard work.
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import jlqml

RowLayout {
    spacing: 0
    Label { text: "Recording"; Layout.preferredWidth: 90 }
    Repeater {
        model: [{ key: "planar", text: "One camera (planar)" },
                { key: "stereo", text: "Two cameras (stereo)" }]
        Button {
            text: modelData.text
            // highlighted rather than checkable: a declined switch keeps the old choice
            highlighted: app.modality === modelData.key
            onClicked: if (app.modality !== modelData.key) Julia.hh_set_modality(modelData.key)
        }
    }
}
