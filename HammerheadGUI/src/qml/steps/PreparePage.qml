// Prepare step: sub-pages for preprocessing, mask, region of interest and
// physical scale. Each sub-page edits the settings through Julia; clicks on
// the viewer go to the open sub-page. A stereo window has no Region page,
// draws the mask on the dewarped grid, and its scale has no pixel size.
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import jlqml
import "prepare"

StepPage {
    id: preparePage
    property bool stereo: false

    title: "Prepare"
    guidance: stereo
        ? "Get the images ready for correlation: preprocessing (applied to each camera's raw " +
          "frames before dewarping), a mask on the dewarped grid over regions without valid " +
          "particles, and the time scale. The viewer shows the shown camera's frame dewarped; " +
          "the shaded border is outside one camera's view. Click the viewer to work on the " +
          "image; drag to zoom, right-drag to pan."
        : "Get the images ready for correlation: preprocessing, a mask over regions " +
          "without valid particles, the region to analyze, and the physical scale. " +
          "Click the viewer to work on the image; drag to zoom, right-drag to pan."

    RowLayout {
        spacing: 0
        ButtonGroup { id: pageGroup }
        Repeater {
            model: stereo
                ? [{ key: "preprocess", text: "Preprocess" }, { key: "mask", text: "Mask" },
                   { key: "scale", text: "Scale" }]
                : [{ key: "preprocess", text: "Preprocess" }, { key: "mask", text: "Mask" },
                   { key: "roi", text: "Region" }, { key: "scale", text: "Scale" }]
            Button {
                text: modelData.text
                checkable: true
                checked: app.preparePage === modelData.key
                ButtonGroup.group: pageGroup
                onClicked: Julia.hh_set_prepare_page(modelData.key)
            }
        }
    }

    PreprocessPane { visible: app.preparePage === "preprocess"; Layout.fillWidth: true }
    MaskPane { visible: app.preparePage === "mask"; Layout.fillWidth: true }
    RoiPane { visible: !stereo && app.preparePage === "roi"; Layout.fillWidth: true }
    ScalePane { visible: app.preparePage === "scale"; stereo: preparePage.stereo; Layout.fillWidth: true }
}
