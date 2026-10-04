// Planar PIV workflow window. All state lives in Julia: `app` (property map),
// `stepModel` and `passModel` (item models); actions call Julia.hh_* functions.
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window
import QtQuick.Dialogs
import Makie
import jlqml
import "steps"

ApplicationWindow {
    id: win
    title: app.title
    width: 1480
    height: 920
    minimumWidth: 1100
    minimumHeight: 680
    visible: true

    property bool quitting: false
    readonly property var stepKeys: ["images", "prepare", "passes", "test", "run", "results"]

    // Closing the main window ends the session (the pop-out window included).
    onClosing: { quitting = true; Qt.quit() }

    header: ToolBar {
        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 8
            anchors.rightMargin: 8
            spacing: 4
            ToolButton { text: "Open settings…"; onClicked: openSettingsDialog.open() }
            ToolButton { text: "Save settings…"; onClicked: saveSettingsDialog.open() }
            Item { Layout.fillWidth: true }
            Label {
                text: app.status
                elide: Text.ElideRight
                Layout.maximumWidth: 560
                opacity: 0.75
            }
            ToolSeparator {}
            ToolButton {
                text: app.popped ? "Dock viewer" : "Pop out viewer"
                onClicked: Julia.hh_toggle_popout()
            }
        }
    }

    footer: PairBar {}

    SplitView {
        anchors.fill: parent
        orientation: Qt.Horizontal

        Pane {
            SplitView.preferredWidth: 760
            SplitView.minimumWidth: 600
            padding: 0
            clip: true

            RowLayout {
                anchors.fill: parent
                spacing: 0

                ListView {
                    id: rail
                    Layout.preferredWidth: 200
                    Layout.fillHeight: true
                    Layout.topMargin: 8
                    model: stepModel
                    interactive: false
                    spacing: 2
                    delegate: StepDelegate {
                        width: rail.width
                        stepNumber: index + 1
                        current: model.key === app.step
                    }
                }

                ToolSeparator { orientation: Qt.Vertical; Layout.fillHeight: true }

                StackLayout {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    // pages scroll; their content must not widen the pane
                    Layout.preferredWidth: 0
                    Layout.minimumWidth: 0
                    currentIndex: Math.max(0, win.stepKeys.indexOf(app.step))
                    ImagesPage {}
                    PreparePage {}
                    PassesPage {}
                    TestPage {}
                    RunPage {}
                    ResultsPage {}
                }
            }
        }

        // The canvas items live as long as their windows: QMLMakie canvases
        // must not be destroyed while the application runs.
        Item {
            SplitView.fillWidth: true
            SplitView.minimumWidth: 400
            MakieArea {
                id: mainCanvas
                anchors.fill: parent
                scene: app.main
                visible: !app.popped
            }
            Label {
                anchors.centerIn: parent
                visible: app.popped
                text: "The viewer is in its own window"
                opacity: 0.7
            }
        }
    }

    Window {
        id: popWindow
        title: "Hammerhead viewer"
        width: 1000
        height: 900
        visible: app.popped
        // Closing the pop-out docks the viewer, except while the app quits
        // (a rejected close would cancel Qt's quit).
        onClosing: (close) => {
            if (!win.quitting) {
                close.accepted = false
                Julia.hh_toggle_popout()
            }
        }
        MakieArea {
            id: popCanvas
            anchors.fill: parent
            scene: app.pop
        }
    }

    Timer {
        interval: 16
        running: true
        repeat: true
        onTriggered: {
            const r = Julia.hh_tick()
            if (r === 2) {
                win.close()
            } else if (r === 1) {
                mainCanvas.update()
                popCanvas.update()
            }
        }
    }

    FileDialog {
        id: openSettingsDialog
        title: "Open settings"
        nameFilters: ["Settings or results (*.jld2)", "All files (*)"]
        onAccepted: Julia.hh_open_settings(selectedFile.toString())
    }
    FileDialog {
        id: saveSettingsDialog
        title: "Save settings"
        fileMode: FileDialog.SaveFile
        defaultSuffix: "jld2"
        nameFilters: ["Settings (*.jld2)"]
        onAccepted: Julia.hh_save_settings(selectedFile.toString())
    }
}
