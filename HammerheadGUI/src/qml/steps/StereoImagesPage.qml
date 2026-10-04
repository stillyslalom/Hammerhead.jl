// Stereo Images step: each camera's frames (entry i of both cameras is the
// same instant), one pairing rule and one representative pair for both.
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import jlqml

StepPage {
    id: page
    title: "Images"
    guidance: "Add each camera's frames in acquisition order — frame i of camera 1 and of " +
              "camera 2 must show the same instant — then choose how they form pairs. Step " +
              "through pairs, frames A and B, and the cameras below the viewer."

    property int addingCamera: 1

    Repeater {
        model: [1, 2]
        ColumnLayout {
            Layout.fillWidth: true
            spacing: 6
            readonly property int camera: modelData
            Label {
                text: "Camera " + camera
                font.weight: Font.DemiBold
                Layout.topMargin: camera === 1 ? 0 : 8
            }
            RowLayout {
                spacing: 8
                Button {
                    text: "Add frames…"
                    highlighted: app["camera" + camera + "Frames"] === 0
                    onClicked: { page.addingCamera = camera; addFramesDialog.open() }
                }
                Button {
                    text: "Clear"
                    enabled: app["camera" + camera + "Frames"] > 0
                    onClicked: Julia.hh_clear_camera_files(camera)
                }
                Label {
                    text: app["camera" + camera + "Summary"]
                    elide: Text.ElideRight
                    Layout.fillWidth: true
                    Layout.leftMargin: 4
                }
            }
            Label {
                text: app["camera" + camera + "Problem"]
                visible: text !== "" && app["camera" + camera + "Frames"] > 0
                color: "#c42b1c"
                wrapMode: Text.WordWrap
                Layout.fillWidth: true
            }
        }
    }

    Label { text: "Pairing"; font.weight: Font.DemiBold; Layout.topMargin: 8 }
    ComboBox {
        Layout.preferredWidth: 280
        textRole: "text"
        valueRole: "value"
        model: [{ text: "Paired: 1–2, 3–4, 5–6 …", value: "paired" },
                { text: "Chained: 1–2, 2–3, 3–4 …", value: "chained" }]
        currentIndex: app.pairMode === "chained" ? 1 : 0
        onActivated: Julia.hh_set_pair_mode(currentValue)
    }

    Label { text: app.framesSummary; Layout.topMargin: 8 }
    Label {
        text: app.framesProblem
        visible: text !== ""
        color: "#c42b1c"
        wrapMode: Text.WordWrap
        Layout.fillWidth: true
    }

    FileDialog {
        id: addFramesDialog
        title: "Add camera " + page.addingCamera + " frames"
        fileMode: FileDialog.OpenFiles
        nameFilters: ["Images (*.tif *.tiff *.png *.bmp *.jpg *.jpeg)", "All files (*)"]
        onAccepted: Julia.hh_add_camera_files(page.addingCamera,
                                              selectedFiles.map(u => u.toString()).join("\n"))
    }
}
