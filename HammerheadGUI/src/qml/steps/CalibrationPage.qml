// Stereo Calibration step: each camera's plate images with their z, the
// dot-grid detection settings and camera model, the fit, the common dewarp
// grid, and self-calibration. The viewer shows the selected plate with its
// detected dots and magnified reprojection residuals.
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import jlqml

StepPage {
    id: page
    title: "Calibration"
    guidance: "Fit each camera from its calibration plate images, build the common dewarp " +
              "grid, then self-calibrate onto the light sheet. Click a plate to show it: dots " +
              "are colored by reprojection error, and the residual arrows are magnified by " +
              "the factor in the viewer's title. A saved calibration opens instead of fitting."

    RowLayout {
        spacing: 8
        Button {
            text: "Open calibration…"
            onClicked: openCalibrationDialog.open()
            ToolTip.visible: hovered
            ToolTip.text: "A saved calibration, or the one stored with stereo results"
        }
        Button {
            text: "Save calibration…"
            enabled: app.hasDewarpers
            onClicked: saveCalibrationDialog.open()
            ToolTip.visible: hovered
            ToolTip.text: "Both cameras (with an applied self-calibration) and the dewarp grid"
        }
    }

    readonly property var plateModel: app.camera === 2 ? plates2Model : plates1Model
    readonly property string sub: app.calibrationPage

    RowLayout {
        spacing: 0
        ButtonGroup { id: subGroup }
        Repeater {
            model: [{ key: "plates", text: "Plates" }, { key: "detection", text: "Detection" },
                    { key: "grid", text: "Dewarp grid" }, { key: "selfcal", text: "Self-calibration" }]
            Button {
                text: modelData.text
                checkable: true
                checked: page.sub === modelData.key
                ButtonGroup.group: subGroup
                onClicked: Julia.hh_set_calibration_page(modelData.key)
            }
        }
    }

    // ---------------------------------------------------------------- plates
    ColumnLayout {
        visible: page.sub === "plates"
        spacing: 10
        Layout.fillWidth: true
        RowLayout {
            spacing: 0
            ButtonGroup { id: cameraGroup }
            Repeater {
                model: [1, 2]
                Button {
                    text: "Camera " + modelData + " (" + app["calPlates" + modelData] + ")"
                    checkable: true
                    checked: app.camera === modelData
                    ButtonGroup.group: cameraGroup
                    onClicked: Julia.hh_set_camera(modelData)
                }
            }
        }
        GridLayout {
            columns: 4
            columnSpacing: 8
            rowSpacing: 4
            Layout.fillWidth: true
            visible: app["calPlates" + app.camera] > 0

            Label { text: "Plate"; opacity: 0.75 }
            Label { text: "z (" + app.calLengthUnit + ")"; opacity: 0.75 }
            Label { text: "Detected"; opacity: 0.75; Layout.fillWidth: true }
            Label { text: "" }

            Repeater {
                model: page.plateModel
                delegate: Button {
                    Layout.row: model.number
                    Layout.column: 0
                    Layout.preferredWidth: 150
                    text: model.number + ". " + model.label
                    flat: model.number !== app.calPlane
                    highlighted: model.number === app.calPlane
                    onClicked: Julia.hh_select_plate(app.camera, model.number)
                    contentItem: Label {
                        text: parent.text
                        elide: Text.ElideMiddle
                        horizontalAlignment: Text.AlignLeft
                    }
                    ToolTip.visible: hovered
                    ToolTip.text: "Show this plate on the viewer"
                }
            }
            Repeater {
                model: page.plateModel
                delegate: TextField {
                    Layout.row: model.number
                    Layout.column: 1
                    Layout.preferredWidth: 80
                    text: model.z
                    selectByMouse: true
                    onEditingFinished: if (text !== model.z) Julia.hh_set_plate_z(app.camera, model.number, text)
                }
            }
            Repeater {
                model: page.plateModel
                delegate: Label {
                    Layout.row: model.number
                    Layout.column: 2
                    Layout.fillWidth: true
                    text: model.info === "" ? "not fitted" : model.info
                    elide: Text.ElideRight
                    opacity: model.info === "" ? 0.6 : 1
                }
            }
            Repeater {
                model: page.plateModel
                delegate: ToolButton {
                    Layout.row: model.number
                    Layout.column: 3
                    text: "✕"
                    onClicked: Julia.hh_remove_plate(app.camera, model.number)
                    ToolTip.visible: hovered
                    ToolTip.text: "Remove this plate"
                }
            }
        }
        RowLayout {
            spacing: 8
            Button {
                text: "Add plate images…"
                highlighted: app["calPlates" + app.camera] === 0
                onClicked: addPlatesDialog.open()
            }
            Button {
                text: "Clear"
                enabled: app["calPlates" + app.camera] > 0
                onClicked: Julia.hh_clear_plates(app.camera)
            }
            Label {
                text: "New plates start at z = 0: enter each plate's z."
                opacity: 0.7
                wrapMode: Text.WordWrap
                Layout.fillWidth: true
            }
        }
    }

    // ---------------------------------------------------------------- detection
    ColumnLayout {
        visible: page.sub === "detection" || page.sub === "plates"
        spacing: 10
        Layout.fillWidth: true
        GridLayout {
            visible: page.sub === "detection"
            columns: 3
            columnSpacing: 10
            rowSpacing: 6

            Label { text: "Dot spacing" }
            TextField {
                Layout.preferredWidth: 110
                text: app.calSpacing
                placeholderText: "e.g. 15"
                onEditingFinished: if (text !== app.calSpacing) Julia.hh_calibration_option("spacing", text)
            }
            Label { text: app.calLengthUnit; opacity: 0.75 }

            Label { text: "Origin offset (x, y)" }
            TextField {
                Layout.preferredWidth: 110
                text: app.calOriginOffset
                placeholderText: "none"
                onEditingFinished: if (text !== app.calOriginOffset) Julia.hh_calibration_option("origin_offset", text)
                ToolTip.visible: hovered
                ToolTip.text: "World position of the dot at the square marker, from the origin dot"
            }
            Label { text: app.calLengthUnit; opacity: 0.75 }

            Label { text: "Two-level plate" }
            Switch {
                checked: app.calTwoLevel
                onToggled: Julia.hh_calibration_option("two_level", checked)
            }
            Label { text: "" }

            Label { text: "Level separation"; enabled: app.calTwoLevel }
            TextField {
                Layout.preferredWidth: 110
                enabled: app.calTwoLevel
                text: app.calLevelSeparation
                onEditingFinished: if (text !== app.calLevelSeparation) Julia.hh_calibration_option("level_separation", text)
            }
            Label { text: app.calLengthUnit; opacity: 0.75; enabled: app.calTwoLevel }

            Label { text: "Dark dots on a bright plate" }
            Switch {
                checked: app.calInvert
                onToggled: Julia.hh_calibration_option("invert", checked)
            }
            Label { text: "" }

            Label { text: "Orientation" }
            ComboBox {
                Layout.preferredWidth: 220
                textRole: "text"; valueRole: "value"
                model: [{ text: "Upright camera", value: "image" },
                        { text: "From the fiducial markers", value: "fiducials" }]
                currentIndex: app.calOrientation === "fiducials" ? 1 : 0
                onActivated: Julia.hh_calibration_option("orientation", currentValue)
            }
            Label { text: "" }

            Label { text: "World length unit" }
            TextField {
                Layout.preferredWidth: 110
                text: app.calLengthUnit
                onEditingFinished: if (text !== app.calLengthUnit) Julia.hh_calibration_option("length_unit", text)
            }
            Label { text: "" }

            Label { text: "Camera model" }
            ComboBox {
                Layout.preferredWidth: 220
                textRole: "text"; valueRole: "value"
                model: [{ text: "Soloff polynomial", value: "soloff" },
                        { text: "Pinhole", value: "pinhole" }]
                currentIndex: app.calModel === "pinhole" ? 1 : 0
                onActivated: Julia.hh_calibration_option("model", currentValue)
            }
            Label { text: "" }
        }
        Label {
            text: app.calError
            visible: text !== ""
            color: "#c42b1c"
            wrapMode: Text.WordWrap
            Layout.fillWidth: true
        }
        RowLayout {
            spacing: 12
            Button {
                text: "Fit cameras"
                highlighted: !app.hasDewarpers || app.calFitStale
                enabled: !app.calFitting && app.calPlates1 > 0 && app.calPlates2 > 0
                onClicked: Julia.hh_fit_calibration()
            }
            BusyIndicator {
                running: app.calFitting
                visible: running
                implicitWidth: 32
                implicitHeight: 32
            }
        }
        Label {
            text: "Plates or detection settings changed since the fit; fit again"
            visible: app.calFitStale
            color: "#b06f00"
            wrapMode: Text.WordWrap
            Layout.fillWidth: true
        }
        Label {
            text: app.calFitStatus
            visible: text !== ""
            wrapMode: Text.WordWrap
            Layout.fillWidth: true
        }
    }

    // ---------------------------------------------------------------- grid
    ColumnLayout {
        visible: page.sub === "grid"
        spacing: 10
        Layout.fillWidth: true
        GridLayout {
            columns: 3
            columnSpacing: 10
            rowSpacing: 6
            enabled: app.calCanBuild

            Label { text: "Coverage" }
            ComboBox {
                Layout.preferredWidth: 220
                textRole: "text"; valueRole: "value"
                model: [{ text: "Seen by both cameras", value: "intersection" },
                        { text: "Seen by either camera", value: "union" }]
                currentIndex: app.calCoverage === "union" ? 1 : 0
                onActivated: Julia.hh_calibration_option("coverage", currentValue)
            }
            Label { text: "" }

            Label { text: "Node spacing" }
            TextField {
                Layout.preferredWidth: 110
                text: app.calGridSpacing
                onEditingFinished: if (text !== app.calGridSpacing) Julia.hh_calibration_option("grid_spacing", text)
                ToolTip.visible: hovered
                ToolTip.text: "\"auto\" matches the cameras' pixel size"
            }
            Label { text: app.calLengthUnit + " (or auto)"; opacity: 0.75 }

            Label { text: "Plane z" }
            TextField {
                Layout.preferredWidth: 110
                text: app.calGridZ
                onEditingFinished: if (text !== app.calGridZ) Julia.hh_calibration_option("grid_z", text)
            }
            Label { text: app.calLengthUnit; opacity: 0.75 }

            Label { text: "Margin" }
            TextField {
                Layout.preferredWidth: 110
                text: app.calMargin
                onEditingFinished: if (text !== app.calMargin) Julia.hh_calibration_option("margin", text)
            }
            Label { text: app.calLengthUnit; opacity: 0.75 }
        }
        RowLayout {
            spacing: 12
            Button {
                text: "Build grid"
                enabled: app.calCanBuild && !app.calBuilding
                onClicked: Julia.hh_build_dewarpers()
            }
            BusyIndicator {
                running: app.calBuilding
                visible: running
                implicitWidth: 32
                implicitHeight: 32
            }
            Label {
                text: app.calGridStatus
                wrapMode: Text.WordWrap
                Layout.fillWidth: true
            }
        }
    }

    // ---------------------------------------------------------------- self-calibration
    ColumnLayout {
        visible: page.sub === "selfcal"
        spacing: 10
        Layout.fillWidth: true
        Label {
            text: "Correlates the two cameras' dewarped images of the same instant and moves the " +
                  "world frame onto the light sheet (Wieneke 2005). Needs the dewarp grid and the " +
                  "particle frames of both cameras. The viewer shows the disparity map: arrows from " +
                  "camera 1 to camera 2, which shrink once the correction is applied."
            wrapMode: Text.WordWrap
            opacity: 0.75
            Layout.fillWidth: true
        }
        RowLayout {
            spacing: 8
            enabled: app.hasDewarpers
            Label { text: "Use the first" }
            SpinBox {
                from: 1; to: 1000; editable: true
                value: app.calSelfcalPairs
                Layout.preferredWidth: 110
                onValueModified: Julia.hh_calibration_option("selfcal_pairs", value)
            }
            Label { text: "pairs" }
            CheckBox {
                text: "Keep disparity maps"
                checked: app.calKeepMaps
                onToggled: Julia.hh_calibration_option("keep_disparity_maps", checked)
            }
        }
        RowLayout {
            spacing: 8
            Button {
                text: "Self-calibrate"
                enabled: app.hasDewarpers && !app.calSelfcalRunning && app.framesProblem === ""
                onClicked: Julia.hh_start_selfcal()
            }
            Button {
                text: "Apply correction"
                enabled: app.hasSelfcal && !app.calSelfcalApplied && !app.calSelfcalRunning
                onClicked: Julia.hh_apply_selfcal()
            }
            BusyIndicator {
                running: app.calSelfcalRunning
                visible: running
                implicitWidth: 32
                implicitHeight: 32
            }
            Label {
                text: app.calSelfcalApplied ? "applied" : ""
                color: "#2e9e44"
            }
        }
        Label {
            text: app.calSelfcalStatus
            visible: text !== ""
            wrapMode: Text.WordWrap
            Layout.fillWidth: true
        }
        RowLayout {
            spacing: 8
            visible: app.calHasMaps
            Label { text: "Viewer: disparity of pass" }
            SpinBox {
                from: 1; to: Math.max(app.calSelfcalPasses, 1); editable: true
                value: app.calDisparityPass
                Layout.preferredWidth: 110
                onValueModified: Julia.hh_set_disparity_pass(value)
            }
            Label {
                text: "of " + app.calSelfcalPasses + " (one arrow scale for every pass)"
                opacity: 0.75
            }
        }
        Frame {
            visible: app.calSelfcalReport !== ""
            Layout.fillWidth: true
            Label {
                anchors.left: parent.left
                anchors.right: parent.right
                text: app.calSelfcalReport
                wrapMode: Text.WordWrap
                lineHeight: 1.2
            }
        }
    }

    FileDialog {
        id: openCalibrationDialog
        title: "Open calibration"
        nameFilters: ["Calibration or stereo results (*.jld2)", "All files (*)"]
        onAccepted: Julia.hh_open_calibration(selectedFile.toString())
    }
    FileDialog {
        id: saveCalibrationDialog
        title: "Save calibration"
        fileMode: FileDialog.SaveFile
        defaultSuffix: "jld2"
        nameFilters: ["Calibration (*.jld2)"]
        onAccepted: Julia.hh_save_calibration(selectedFile.toString())
    }
    FileDialog {
        id: addPlatesDialog
        title: "Add camera " + app.camera + " calibration plates"
        fileMode: FileDialog.OpenFiles
        nameFilters: ["Images (*.tif *.tiff *.png *.bmp *.jpg *.jpeg)", "All files (*)"]
        onAccepted: Julia.hh_add_plates(app.camera, selectedFiles.map(u => u.toString()).join("\n"))
    }
}
