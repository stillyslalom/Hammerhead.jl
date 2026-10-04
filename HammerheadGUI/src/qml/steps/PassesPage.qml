import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import jlqml

StepPage {
    id: passesPage
    // the window's planar analysis modes; the stereo window has PIV only
    property bool particleModes: false
    readonly property bool particles: app.particleMode === true

    title: particles ? "Particles" : "Passes"
    guidance: particles
        ? "Detect particles, match them between the frames of a pair, and validate the " +
          "matches. The viewer circles the particles detected on the shown frame with these " +
          "settings. The PIV predictor below centers each particle's search on the local flow."
        : "Start from a preset, then adjust. Each pass refines the previous one; the first " +
          "window should be at least four times the largest displacement. The viewer " +
          "outlines each window size against the particles."

    Label { text: "Analysis"; font.weight: Font.DemiBold; visible: passesPage.particleModes }
    ComboBox {
        visible: passesPage.particleModes
        Layout.preferredWidth: 340
        textRole: "text"; valueRole: "value"
        model: [{ text: "PIV, per pair (time series)", value: "sequence" },
                { text: "PIV ensemble (one mean field)", value: "ensemble" },
                { text: "PTV: particle matches per pair", value: "ptv" },
                { text: "Particle tracking through the frames", value: "tracking" }]
        currentIndex: Math.max(0, ["sequence", "ensemble", "ptv", "tracking"].indexOf(app.mode))
        onActivated: Julia.hh_set_mode(currentValue)
    }
    Label {
        visible: passesPage.particleModes && app.mode === "tracking"
        text: "Tracking follows every frame in the order listed on the Images step (a " +
              "time-resolved recording); the pairing there sets only the representative pair."
        opacity: 0.75
        wrapMode: Text.WordWrap
        Layout.fillWidth: true
    }

    // ---------------------------------------------------------------- particles
    Label { text: "Detection"; font.weight: Font.DemiBold; visible: particles; Layout.topMargin: 8 }
    GridLayout {
        visible: particles
        columns: 3
        columnSpacing: 10
        rowSpacing: 6
        Label { text: "Threshold" }
        TextField {
            Layout.preferredWidth: 100
            text: app.ptvThreshold
            onEditingFinished: if (text !== app.ptvThreshold) Julia.hh_particle_option("threshold", text)
            ToolTip.visible: hovered
            ToolTip.text: "\"auto\": median + k × noise of the frame; or an intensity"
        }
        Label { text: "intensity, or auto"; opacity: 0.75 }
        Label { text: "Auto threshold k" }
        TextField {
            Layout.preferredWidth: 100
            enabled: app.ptvThreshold === "auto"
            text: app.ptvThresholdK
            onEditingFinished: if (text !== app.ptvThresholdK) Julia.hh_particle_option("threshold_k", text)
        }
        Label { text: "× noise"; opacity: 0.75 }
        Label { text: "Minimum separation" }
        TextField {
            Layout.preferredWidth: 100
            text: app.ptvMinSeparation
            onEditingFinished: if (text !== app.ptvMinSeparation) Julia.hh_particle_option("min_separation", text)
        }
        Label { text: "px"; opacity: 0.75 }
        Label { text: "Diameter" }
        RowLayout {
            TextField {
                Layout.preferredWidth: 60
                text: app.ptvMinDiameter
                onEditingFinished: if (text !== app.ptvMinDiameter) Julia.hh_particle_option("min_diameter", text)
            }
            Label { text: "to" }
            TextField {
                Layout.preferredWidth: 60
                text: app.ptvMaxDiameter
                onEditingFinished: if (text !== app.ptvMaxDiameter) Julia.hh_particle_option("max_diameter", text)
            }
        }
        Label { text: "px"; opacity: 0.75 }
    }
    Label {
        visible: particles && app.ptvDetectStatus !== ""
        text: app.ptvDetectStatus
        wrapMode: Text.WordWrap
        Layout.fillWidth: true
    }

    Label { text: "Matching"; font.weight: Font.DemiBold; visible: particles; Layout.topMargin: 8 }
    GridLayout {
        visible: particles
        columns: 3
        columnSpacing: 10
        rowSpacing: 6
        Label { text: "Search radius" }
        TextField {
            Layout.preferredWidth: 100
            text: app.ptvSearchRadius
            onEditingFinished: if (text !== app.ptvSearchRadius) Julia.hh_particle_option("search_radius", text)
            ToolTip.visible: hovered
            ToolTip.text: "Around each particle's predicted position in frame B"
        }
        Label { text: "px"; opacity: 0.75 }
        Label { text: "PIV predictor" }
        Switch {
            checked: app.ptvPredictor === "piv"
            onToggled: Julia.hh_particle_option("predictor", checked ? "piv" : "none")
            ToolTip.visible: hovered
            ToolTip.text: "Off: search around each particle's own position (small displacements)"
        }
        Label { text: "" }
        Label { text: "Intensity weight" }
        TextField {
            Layout.preferredWidth: 100
            text: app.ptvIntensityWeight
            onEditingFinished: if (text !== app.ptvIntensityWeight) Julia.hh_particle_option("intensity_weight", text)
        }
        Label { text: "0 = distance only"; opacity: 0.75 }
        Label { text: "Diameter weight" }
        TextField {
            Layout.preferredWidth: 100
            text: app.ptvDiameterWeight
            onEditingFinished: if (text !== app.ptvDiameterWeight) Julia.hh_particle_option("diameter_weight", text)
        }
        Label { text: "" }
    }

    Label { text: "Validation"; font.weight: Font.DemiBold; visible: particles; Layout.topMargin: 8 }
    GridLayout {
        visible: particles
        columns: 3
        columnSpacing: 10
        rowSpacing: 6
        Label { text: "Flag outlier matches" }
        Switch {
            checked: app.ptvUodEnable
            onToggled: Julia.hh_particle_option("uod_enable", checked)
            ToolTip.visible: hovered
            ToolTip.text: "Normalized median test against neighboring matches; flagged, never replaced"
        }
        Label { text: "" }
        Label { text: "Threshold"; enabled: app.ptvUodEnable }
        TextField {
            Layout.preferredWidth: 100
            enabled: app.ptvUodEnable
            text: app.ptvUodThreshold
            onEditingFinished: if (text !== app.ptvUodThreshold) Julia.hh_particle_option("uod_threshold", text)
        }
        Label { text: "" }
        Label { text: "Neighbors"; enabled: app.ptvUodEnable }
        TextField {
            Layout.preferredWidth: 100
            enabled: app.ptvUodEnable
            text: app.ptvUodNeighbors
            onEditingFinished: if (text !== app.ptvUodNeighbors) Julia.hh_particle_option("uod_neighbors", text)
        }
        Label { text: "" }
        Label { text: "Noise floor ε"; enabled: app.ptvUodEnable }
        TextField {
            Layout.preferredWidth: 100
            enabled: app.ptvUodEnable
            text: app.ptvUodEpsilon
            onEditingFinished: if (text !== app.ptvUodEpsilon) Julia.hh_particle_option("uod_epsilon", text)
        }
        Label { text: "px"; opacity: 0.75 }
    }

    Label { text: "Tracks"; font.weight: Font.DemiBold; visible: particles && app.mode === "tracking"; Layout.topMargin: 8 }
    GridLayout {
        visible: particles && app.mode === "tracking"
        columns: 3
        columnSpacing: 10
        rowSpacing: 6
        Label { text: "Shortest track kept" }
        SpinBox {
            from: 2; to: 1000; editable: true
            value: app.ptvMinTrackLength
            Layout.preferredWidth: 110
            onValueModified: Julia.hh_particle_option("min_track_length", value)
        }
        Label { text: "frames"; opacity: 0.75 }
        Label { text: "Bridge gaps of up to" }
        SpinBox {
            from: 0; to: 100; editable: true
            value: app.ptvMaxGap
            Layout.preferredWidth: 110
            onValueModified: Julia.hh_particle_option("max_gap", value)
        }
        Label { text: "missed frames"; opacity: 0.75 }
    }
    Label {
        text: app.ptvError
        visible: particles && text !== ""
        color: "#c42b1c"
        wrapMode: Text.WordWrap
        Layout.fillWidth: true
    }

    Label {
        text: "PIV predictor passes"
        font.weight: Font.DemiBold
        visible: particles && app.ptvPredictor === "piv"
        Layout.topMargin: 12
    }

    RowLayout {
        visible: !particles || app.ptvPredictor === "piv"
        spacing: 0
        Label { text: "Preset"; Layout.preferredWidth: 90 }
        ButtonGroup { id: presetGroup }
        Repeater {
            model: [{ key: "low", text: "Low" }, { key: "medium", text: "Medium" },
                    { key: "high", text: "High" }]
            Button {
                text: modelData.text
                checkable: true
                checked: app.preset === modelData.key
                ButtonGroup.group: presetGroup
                onClicked: Julia.hh_fill_preset(modelData.key)
            }
        }
        Label {
            text: app.preset === "custom" ? "custom" : ""
            opacity: 0.7
            Layout.leftMargin: 12
        }
    }

    // pass table
    GridLayout {
        visible: !particles || app.ptvPredictor === "piv"
        columns: 6
        columnSpacing: 8
        rowSpacing: 6
        Layout.topMargin: 8

        Label { text: "" }
        Label { text: "Window (px)"; opacity: 0.75 }
        Label { text: "Search (px)"; opacity: 0.75 }
        Label { text: "Overlap (%)"; opacity: 0.75 }
        Label { text: "Repeats"; opacity: 0.75 }
        Label { text: "" }

        Repeater {
            model: passModel
            delegate: Label {
                Layout.row: model.number
                Layout.column: 0
                text: model.number
            }
        }
        Repeater {
            model: passModel
            delegate: SpinBox {
                Layout.row: model.number
                Layout.column: 1
                Layout.preferredWidth: 100
                from: 4; to: 2048; stepSize: 8; editable: true
                value: model.window
                onValueModified: Julia.hh_set_pass(model.number, "window", value)
            }
        }
        Repeater {
            model: passModel
            delegate: SpinBox {
                Layout.row: model.number
                Layout.column: 2
                Layout.preferredWidth: 100
                from: 4; to: 2048; stepSize: 8; editable: true
                value: model.search
                onValueModified: Julia.hh_set_pass(model.number, "search", value)
            }
        }
        Repeater {
            model: passModel
            delegate: SpinBox {
                Layout.row: model.number
                Layout.column: 3
                Layout.preferredWidth: 92
                from: 0; to: 90; stepSize: 5; editable: true
                value: Math.round(model.overlap)
                onValueModified: Julia.hh_set_pass(model.number, "overlap", value)
            }
        }
        Repeater {
            model: passModel
            delegate: SpinBox {
                Layout.row: model.number
                Layout.column: 4
                Layout.preferredWidth: 100
                from: 1; to: 10; editable: true
                enabled: app.mode !== "ensemble"
                ToolTip.visible: hovered && app.mode === "ensemble"
                ToolTip.text: "An ensemble runs each pass once; add a pass to repeat a window size"
                value: model.iterations
                onValueModified: Julia.hh_set_pass(model.number, "iterations", value)
            }
        }
        Repeater {
            model: passModel
            delegate: ToolButton {
                Layout.row: model.number
                Layout.column: 5
                text: "✕"
                enabled: passModel.rowCount() > 1
                ToolTip.visible: hovered
                ToolTip.text: "Remove pass " + model.number
                onClicked: Julia.hh_remove_pass(model.number)
            }
        }
    }

    RowLayout {
        visible: !particles || app.ptvPredictor === "piv"
        Button { text: "Add pass"; onClicked: Julia.hh_add_pass() }
        Label { text: app.passesSummary; opacity: 0.75; Layout.leftMargin: 8 }
    }
    Label {
        text: app.passesError
        visible: text !== ""
        color: "#c42b1c"
        wrapMode: Text.WordWrap
        Layout.fillWidth: true
    }

    Label {
        text: "Correlation"
        font.weight: Font.DemiBold
        Layout.topMargin: 12
        visible: !particles || app.ptvPredictor === "piv"
    }
    GridLayout {
        visible: !particles || app.ptvPredictor === "piv"
        columns: 2
        columnSpacing: 16
        rowSpacing: 8

        Label { text: "Method" }
        ComboBox {
            Layout.preferredWidth: 220
            textRole: "text"; valueRole: "value"
            model: [{ text: "Cross-correlation", value: "cross" },
                    { text: "Phase correlation", value: "phase" }]
            currentIndex: app.correlation === "phase" ? 1 : 0
            onActivated: Julia.hh_set_option("correlation", currentValue)
        }
        Label { text: "Subpixel fit" }
        ComboBox {
            Layout.preferredWidth: 220
            textRole: "text"; valueRole: "value"
            model: [{ text: "3-point Gaussian", value: "gauss3" },
                    { text: "9-point Gaussian", value: "gauss9" },
                    { text: "2-D Gaussian fit", value: "gauss2d" }]
            currentIndex: Math.max(0, ["gauss3", "gauss9", "gauss2d"].indexOf(app.subpixel))
            onActivated: Julia.hh_set_option("subpixel", currentValue)
        }
        Label { text: "Zero padding" }
        Switch {
            checked: app.padding
            onToggled: Julia.hh_set_option("padding", checked)
            ToolTip.visible: hovered
            ToolTip.text: "Removes the circular-correlation bias toward zero displacement; slower. " +
                          "With Gaussian weighting, the most accurate setting (about 0.03 px RMS)"
        }
        Label { text: "Gaussian weighting" }
        Switch {
            checked: app.apodization
            onToggled: Julia.hh_set_option("apodization", checked)
            ToolTip.visible: hovered
            ToolTip.text: "Weights each window toward its center (Gaussian apodization), " +
                          "suppressing edge effects"
        }
        Label { text: "Image interpolation" }
        ComboBox {
            Layout.preferredWidth: 220
            textRole: "text"; valueRole: "value"
            model: [{ text: "Cubic B-spline", value: "cubic" },
                    { text: "Bilinear", value: "linear" }]
            currentIndex: app.imageInterpolation === "linear" ? 1 : 0
            onActivated: Julia.hh_set_option("image_interpolation", currentValue)
            ToolTip.visible: hovered
            ToolTip.text: "How deforming passes resample the images. Bilinear is faster but " +
                          "smooths particle images and adds a sub-pixel bias"
        }
        Label { text: "Predictor interpolation" }
        ComboBox {
            Layout.preferredWidth: 220
            textRole: "text"; valueRole: "value"
            model: [{ text: "Bilinear", value: "linear" },
                    { text: "Cubic B-spline", value: "cubic" }]
            currentIndex: app.predictorInterpolation === "cubic" ? 1 : 0
            onActivated: Julia.hh_set_option("predictor_interpolation", currentValue)
            ToolTip.visible: hovered
            ToolTip.text: "How the previous pass's vectors are interpolated to deform the images. " +
                          "Cubic follows strongly curved flow more closely"
        }
        Label { text: "Uncertainty on final pass"; visible: !particles }
        Switch {
            visible: !particles
            checked: app.uncertainty
            onToggled: Julia.hh_set_option("uncertainty", checked)
            ToolTip.visible: hovered
            ToolTip.text: "Per-vector random-error estimate (Wieneke 2015); needs a converged final pass"
        }
    }

    Label {
        text: "Validation"
        font.weight: Font.DemiBold
        Layout.topMargin: 12
        visible: !particles
    }
    GridLayout {
        visible: !particles
        columns: 3
        columnSpacing: 16
        rowSpacing: 8
        Label { text: "Normalized median test" }
        Switch {
            checked: app.uodEnable
            onToggled: Julia.hh_set_option("uod_enable", checked)
            ToolTip.visible: hovered
            ToolTip.text: "Flags vectors that differ from their neighbors' median by more than " +
                          "the threshold times the neighbors' median residual"
        }
        Label { text: "" }
        Label { text: "Threshold"; enabled: app.uodEnable }
        TextField {
            Layout.preferredWidth: 100
            enabled: app.uodEnable
            text: app.uodThreshold
            onEditingFinished: if (text !== app.uodThreshold) Julia.hh_set_option("uod_threshold", text)
        }
        Label { text: "2 is typical"; opacity: 0.75 }
        Label { text: "Neighborhood"; enabled: app.uodEnable }
        ComboBox {
            Layout.preferredWidth: 100
            enabled: app.uodEnable
            textRole: "text"; valueRole: "value"
            model: [{ text: "3 × 3", value: 1 }, { text: "5 × 5", value: 2 }, { text: "7 × 7", value: 3 }]
            currentIndex: Math.max(0, [1, 2, 3].indexOf(app.uodNeighborhood))
            onActivated: Julia.hh_set_option("uod_neighborhood", currentValue)
            ToolTip.visible: hovered
            ToolTip.text: "5 × 5 avoids flagging smooth gradients at the field edges"
        }
        Label { text: "" }
        Label { text: "Minimum peak ratio" }
        TextField {
            Layout.preferredWidth: 100
            text: app.minPeakRatio
            onEditingFinished: if (text !== app.minPeakRatio) Julia.hh_set_option("min_peak_ratio", text)
            ToolTip.visible: hovered
            ToolTip.text: "Flags vectors whose highest correlation peak is not this many times the " +
                          "second; 1 disables the check"
        }
        Label { text: "1 = off"; opacity: 0.75 }
        Label { text: "Replace flagged vectors" }
        Switch {
            checked: app.replaceOutliers
            onToggled: Julia.hh_set_option("replace_outliers", checked)
            ToolTip.visible: hovered
            ToolTip.text: "Fill flagged vectors from their valid neighbors (they stay flagged). " +
                          "Intermediate passes always replace them for the predictor."
        }
        Label { text: "" }
    }

    Label {
        text: "Correlation probe"
        font.weight: Font.DemiBold
        Layout.topMargin: 12
        visible: !particles
    }
    Label {
        visible: !particles
        text: app.probeOnPasses
            ? "Click the image to correlate one final-pass window of the processed pair there; " +
              "right-click removes the probe."
            : "Add frames to probe the correlation."
        wrapMode: Text.WordWrap
        opacity: 0.75
        Layout.fillWidth: true
    }
    RowLayout {
        visible: !particles && app.probeSummary !== ""
        Label {
            text: app.probeSummary
            wrapMode: Text.WordWrap
            Layout.fillWidth: true
        }
        Button { text: "Remove probe"; onClicked: Julia.hh_clear_probe() }
    }

    Label { text: "Evaluation"; font.weight: Font.DemiBold; Layout.topMargin: 12 }
    GridLayout {
        columns: 2
        columnSpacing: 16
        rowSpacing: 8
        Label { text: "Mode"; visible: !passesPage.particleModes }
        ComboBox {
            visible: !passesPage.particleModes
            Layout.preferredWidth: 300
            textRole: "text"; valueRole: "value"
            model: [{ text: "Per pair (time series)", value: "sequence" },
                    { text: "Ensemble (one mean field)", value: "ensemble" }]
            currentIndex: app.mode === "ensemble" ? 1 : 0
            onActivated: Julia.hh_set_mode(currentValue)
        }
        Item { width: 1; height: 1; visible: app.mode === "ensemble" }
        Label {
            text: "Sums each window's correlation over all pairs and finds one peak: for " +
                  "steady flow whose single pairs are too noisy. Each pass runs once " +
                  "(repeats are ignored; add a pass to repeat a window size)."
            visible: app.mode === "ensemble"
            opacity: 0.75
            wrapMode: Text.WordWrap
            Layout.preferredWidth: 300
        }
        Label { text: "Precision" }
        ComboBox {
            Layout.preferredWidth: 300
            textRole: "text"; valueRole: "value"
            model: [{ text: "Float64", value: "Float64" },
                    { text: "Float32 (less memory)", value: "Float32" }]
            currentIndex: app.precision === "Float32" ? 1 : 0
            onActivated: Julia.hh_set_precision(currentValue)
        }
        Label { text: "Run on the GPU"; enabled: gpuSwitch.enabled }
        RowLayout {
            spacing: 8
            Switch {
                id: gpuSwitch
                enabled: app.gpuInstalled && !app.gpuLoading && !passesPage.particleModes
                checked: app.gpuOn
                onToggled: {
                    Julia.hh_use_gpu(checked)
                    checked = Qt.binding(() => app.gpuOn)
                }
                ToolTip.visible: hovered
                ToolTip.text: !app.gpuInstalled ?
                    "Install a GPU package (CUDA for NVIDIA, AMDGPU for AMD) in the " +
                    "environment to enable this" :
                    passesPage.particleModes ? "Particle analysis runs on the CPU" :
                    "Tests and runs use " + app.gpuPackages + "; loading it the first " +
                    "time takes a while"
            }
            BusyIndicator { running: app.gpuLoading; visible: running; implicitHeight: 24; implicitWidth: 24 }
            Label {
                text: app.gpuStatus
                visible: text !== ""
                opacity: 0.75
                wrapMode: Text.WordWrap
                Layout.preferredWidth: 240
            }
        }
    }
}
