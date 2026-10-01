/*
  Dashboard charts.

  Plain ES5, no build step, no chart library. A bar chart is a handful of
  <rect> elements, so a library would be more code than this is, plus one
  more dependency to pin and audit.

  Three rules this file follows, matching the rest of the dashboard:

  1. **It computes no statistic.** Every label and value comes from the
     data-labels / data-values attributes the server rendered from
     AnalyticsService. If a number is wrong the bug is upstream, and this
     file must not quietly round or correct it. The only derived values are
     presentational: the axis maximum and the label thinning, neither of
     which is a reported figure.
  2. **It is optional.** The page renders complete without it, because every
     figure is already in the HTML as text and each chart carries a
     <noscript> table. A failure here costs a chart, never a number.
  3. **It is deterministic.** The same data produces byte-identical SVG:
     no animation, no randomness and no dependence on the current time.
*/

(function () {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";

  /* Geometry in user units. The SVG scales through its viewBox, so these
     are proportions rather than pixels and the chart is responsive without
     a resize listener. */
  var WIDTH = 660;
  var HEIGHT = 210;
  var PAD = { top: 22, right: 10, bottom: 32, left: 36 };

  /* Past this many bars the chart stops being readable, so only the most
     recent are drawn. The noscript table still lists them all. */
  var MAX_BARS = 31;

  /* At most this many x-axis labels, whatever the data length. */
  var MAX_X_LABELS = 7;

  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  /* --- Small DOM helpers ------------------------------------------- */
  function el(name, attrs) {
    var node = document.createElementNS(SVG_NS, name);
    for (var key in attrs) {
      if (Object.prototype.hasOwnProperty.call(attrs, key)) {
        node.setAttribute(key, String(attrs[key]));
      }
    }
    return node;
  }

  function text(value, attrs) {
    var node = el("text", attrs);
    node.textContent = value;
    return node;
  }

  /* --- Diagnostics -------------------------------------------------- */
  function warn(message, detail) {
    if (window.console && window.console.warn) {
      window.console.warn("dashboard: " + message, detail || "");
    }
  }

  /* --- Data --------------------------------------------------------- */
  function isArray(value) {
    return Object.prototype.toString.call(value) === "[object Array]";
  }

  /* Read a JSON data attribute, returning null rather than throwing. A
     malformed attribute should leave the caption and the noscript table in
     place, not blank the page. */
  function readJSON(node, attribute) {
    var raw = node.getAttribute(attribute);
    if (!raw) {
      return null;
    }
    try {
      return JSON.parse(raw);
    } catch (error) {
      warn("could not parse " + attribute, error);
      return null;
    }
  }

  /* The pair is usable only when both halves are arrays of equal length.
     Anything else is treated as "no data" rather than half-drawn. */
  function readSeries(container) {
    var labels = readJSON(container, "data-labels");
    var values = readJSON(container, "data-values");

    if (!isArray(labels) || !isArray(values) || labels.length === 0) {
      return null;
    }
    if (labels.length !== values.length) {
      warn("label and value counts differ, chart skipped");
      return null;
    }
    return { labels: labels, values: values };
  }

  /* --- Formatting ---------------------------------------------------- */
  /* "2026-09-26" becomes "26 Sep", short enough to fit without rotating.
     ISO week labels such as "2026-W39" are already compact and pass
     through unchanged. */
  function shortLabel(label) {
    var match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(label));
    if (!match) {
      return String(label);
    }
    return parseInt(match[3], 10) + " " + MONTHS[parseInt(match[2], 10) - 1];
  }

  /* A readable summary for screen readers, because bars are shapes rather
     than text. Built only from values already on the page. */
  function describe(labels, values, name) {
    var total = 0;
    var peak = 0;
    var peakLabel = "";

    for (var i = 0; i < values.length; i += 1) {
      total += values[i];
      if (values[i] > peak) {
        peak = values[i];
        peakLabel = labels[i];
      }
    }
    return name + ": " + total + " commands across " + values.length +
           (values.length === 1 ? " period." : " periods.") +
           " Busiest was " + peakLabel + " with " + peak + ".";
  }

  /* A "nice" axis maximum: the smallest round number at or above the peak,
     so the top gridline is a meaningful value rather than the raw maximum.
     Presentational only; it never touches a reported figure. */
  function niceMax(peak) {
    if (peak <= 5) {
      return peak > 0 ? peak : 1;
    }
    var magnitude = Math.pow(10, Math.floor(Math.log(peak) / Math.LN10));
    var scaled = peak / magnitude;
    var step = scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10;
    return step * magnitude;
  }

  /* --- Rendering ----------------------------------------------------- */
  function drawGrid(svg, plotHeight, axisMax) {
    /* Gridlines at 0, 50 and 100 per cent of the axis. */
    var fractions = [0, 0.5, 1];
    for (var i = 0; i < fractions.length; i += 1) {
      var y = PAD.top + plotHeight * (1 - fractions[i]);
      svg.appendChild(el("line", {
        x1: PAD.left,
        x2: WIDTH - PAD.right,
        y1: y,
        y2: y,
        "class": fractions[i] === 0 ? "baseline" : "grid"
      }));
      svg.appendChild(text(String(Math.round(axisMax * fractions[i])), {
        x: PAD.left - 8,
        y: y + 3,
        "text-anchor": "end",
        "class": "axis-label"
      }));
    }
  }

  function drawBars(svg, labels, values, plotHeight, axisMax, slot) {
    var barWidth = Math.max(2, Math.min(slot * 0.62, 40));
    var baseY = PAD.top + plotHeight;

    /* Show every nth label so the axis stays readable at any width. */
    var stride = Math.max(1, Math.ceil(labels.length / MAX_X_LABELS));
    /* Value labels only when they will not collide with neighbours. */
    var showValues = slot > 26;

    for (var i = 0; i < values.length; i += 1) {
      var count = typeof values[i] === "number" ? values[i] : 0;
      var barHeight = axisMax > 0 ? (count / axisMax) * plotHeight : 0;
      var x = PAD.left + slot * i + (slot - barWidth) / 2;
      var barY = baseY - barHeight;

      if (barHeight > 0) {
        svg.appendChild(el("rect", {
          x: x, y: barY, width: barWidth, height: barHeight,
          "class": "bar"
        }));

        if (showValues && barHeight > 12) {
          svg.appendChild(text(String(count), {
            x: x + barWidth / 2,
            y: barY - 5,
            "text-anchor": "middle",
            "class": "value-label"
          }));
        }
      }

      if (i % stride === 0) {
        svg.appendChild(text(shortLabel(labels[i]), {
          x: x + barWidth / 2,
          y: HEIGHT - PAD.bottom + 15,
          "text-anchor": "middle",
          "class": "axis-label"
        }));
      }
    }
  }

  function render(container, caption) {
    var series = readSeries(container);

    if (!series) {
      /* Nothing usable. Say so in the chart area rather than leaving a blank
         rectangle that reads as a rendering failure. The noscript table and
         every surrounding figure are untouched. */
      container.appendChild(text("No data to chart in this range.", {
        x: WIDTH / 2, y: HEIGHT / 2, "text-anchor": "middle",
        "class": "empty-note"
      }));
      return;
    }

    /* Too many bars to read, so show the most recent. The service returns
       chronological order, so the tail is the latest. */
    var start = Math.max(0, series.values.length - MAX_BARS);
    var labels = series.labels.slice(start);
    var values = series.values.slice(start);

    var peak = 0;
    for (var i = 0; i < values.length; i += 1) {
      if (values[i] > peak) {
        peak = values[i];
      }
    }

    var plotHeight = HEIGHT - PAD.top - PAD.bottom;
    var slot = (WIDTH - PAD.left - PAD.right) / values.length;

    var svg = el("svg", {
      viewBox: "0 0 " + WIDTH + " " + HEIGHT,
      preserveAspectRatio: "xMidYMid meet",
      role: "img",
      "aria-label": describe(labels, values, caption)
    });

    drawGrid(svg, plotHeight, niceMax(peak));
    drawBars(svg, labels, values, plotHeight, niceMax(peak), slot);
    container.appendChild(svg);
  }

  /* --- Entry point --------------------------------------------------- */
  function init() {
    var charts = document.querySelectorAll('[data-chart="bar"]');

    for (var i = 0; i < charts.length; i += 1) {
      var container = charts[i];
      /* The caption is the nearest one, so the accessible label names the
         same thing a sighted reader sees. */
      var captionNode = container.parentNode
        ? container.parentNode.querySelector(".chart__caption")
        : null;
      var caption = captionNode ? captionNode.textContent.trim() : "Activity";

      try {
        render(container, caption);
      } catch (error) {
        /* A chart is decoration. If it fails, the page keeps its numbers. */
        if (window.console && window.console.error) {
          window.console.error("dashboard: chart failed to render", error);
        }
      }
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();