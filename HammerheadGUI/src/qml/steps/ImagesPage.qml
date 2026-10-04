import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import jlqml

StepPage {
    title: "Images"
    guidance: "Add the recording's frames in acquisition order, then choose how they form pairs. " +
              "Step through pairs and switch between frame A and B below the viewer: particles " +
              "should shift slightly between frames, not jump."

    RowLayout {
        spacing: 8
        Button {
            text: "Add frames…"
            highlighted: app.pairCount === 0
            onClicked: addFramesDialog.open()
        }
        Button {
            text: "Clear"
            enabled: app.framesSummary !== "no frames"
            onClicked: Julia.hh_clear_files()
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
        title: "Add frames"
        fileMode: FileDialog.OpenFiles
        nameFilters: ["Images (*.tif *.tiff *.png *.bmp *.jpg *.jpeg)", "All files (*)"]
        onAccepted: Julia.hh_add_files(selectedFiles.map(u => u.toString()).join("\n"))
    }
}
