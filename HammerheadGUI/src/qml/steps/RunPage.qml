import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import jlqml

StepPage {
    title: "Run"
    guidance: "Process every pair with the current settings. Results are written to the output " +
              "file as they finish, together with the settings that produced them; the viewer " +
              "shows the latest pair. Cancelling keeps the pairs already finished."

    Label { text: "Output file"; font.weight: Font.DemiBold }
    RowLayout {
        Layout.fillWidth: true
        TextField {
            Layout.fillWidth: true
            text: app.outputPath
            placeholderText: "Keep results in memory only"
            enabled: !app.runRunning
            onEditingFinished: Julia.hh_set_output(text)
        }
        Button {
            text: "Browse…"
            enabled: !app.runRunning
            onClicked: outputDialog.open()
        }
    }

    RowLayout {
        spacing: 8
        Layout.topMargin: 8
        Button {
            text: "Run " + app.pairCount + " pairs"
            highlighted: true
            enabled: !app.runRunning && app.framesProblem === ""
            onClicked: Julia.hh_start_run()
        }
        Button {
            text: "Cancel"
            enabled: app.runRunning
            onClicked: Julia.hh_cancel_run()
        }
    }

    ProgressBar {
        Layout.fillWidth: true
        from: 0
        to: Math.max(1, app.runTotal)
        value: app.runDone
        visible: app.runRunning || app.runDone > 0
    }
    Label {
        text: app.runRunning ? app.runDone + " of " + app.runTotal + " pairs · " + app.runEta
                             : app.runStatus
        visible: text !== ""
        wrapMode: Text.WordWrap
        Layout.fillWidth: true
    }

    FileDialog {
        id: outputDialog
        title: "Results file"
        fileMode: FileDialog.SaveFile
        defaultSuffix: "jld2"
        nameFilters: ["Results (*.jld2)"]
        onAccepted: Julia.hh_set_output(selectedFile.toString())
    }
}
