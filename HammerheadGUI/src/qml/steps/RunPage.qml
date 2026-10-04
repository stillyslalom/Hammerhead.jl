import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import jlqml

StepPage {
    title: "Run"
    guidance: app.mode === "ensemble"
        ? "Pool the correlations of every pair into one result with the current settings. " +
          "The result is written to the output file when the run finishes, together with the " +
          "settings that produced it."
        : app.mode === "tracking"
        ? "Track particles through every frame in order with the current settings. The tracks " +
          "are written to the output file when the run finishes, together with the settings " +
          "that produced them."
        : "Process every pair with the current settings. Results are written to the output " +
          "file as they finish, together with the settings that produced them; the viewer " +
          "shows the latest pair. Canceling keeps the pairs already finished."

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
        Button {
            text: "Clear"
            enabled: !app.runRunning && app.outputPath !== ""
            onClicked: Julia.hh_set_output("")
            ToolTip.visible: hovered
            ToolTip.text: "Keep results in memory only"
        }
    }

    RowLayout {
        spacing: 8
        Layout.topMargin: 8
        Button {
            text: app.mode === "ensemble" ? "Run ensemble of " + app.pairCount + " pairs" :
                  app.mode === "tracking" ? "Track through " + app.frameCount + " frames" :
                                            "Run " + app.pairCount + " pairs"
            highlighted: true
            enabled: !app.runRunning && app.analysisProblem === ""
            onClicked: Julia.hh_start_run()
        }
        Button {
            text: "Cancel"
            enabled: app.runRunning
            onClicked: Julia.hh_cancel_run()
        }
    }
    Label {
        text: app.mode === "tracking"
            ? "Canceling tracking stops at the next frame and keeps no result."
            : "Canceling an ensemble stops after the pair in flight and keeps no result."
        visible: app.mode === "ensemble" || app.mode === "tracking"
        opacity: 0.75
        wrapMode: Text.WordWrap
        Layout.fillWidth: true
    }

    ProgressBar {
        Layout.fillWidth: true
        from: 0
        to: Math.max(1, app.runTotal)
        value: app.runDone
        visible: app.runRunning || app.runDone > 0
    }
    Label {
        text: app.runRunning ? app.runProgress + (app.runEta === "" ? "" : " · " + app.runEta)
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
