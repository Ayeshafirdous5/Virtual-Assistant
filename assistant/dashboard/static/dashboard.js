/*
  Dashboard charts.

  Plain ES5-compatible script, no build step, no chart library. Each chart is
  a handful of <rect> elements, which is a few lines of DOM work and avoids a
  dependency that would have to be audited and pinned like any other.

  Two rules this file follows, matching the rest of the dashboard:

  1. It never computes a statistic. Every label and value is read from the
     data-labels / data-values attributes that the server rendered from
     AnalyticsService. If a number is wrong the bug is upstream, and this
     file must not quietly correct or round it.
  2. It is optional. The page renders complete without it -- every figure is
     already in the HTML as text, and each chart carries a <noscript> table.
     A failure here costs a chart, never a number.

  Labels are thinned rather than rotated: a date axis with 30 ticks is
  unreadable, and showing every fourth one hides no data, since the
  underlying table lists all of it.
*/

(function () {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";

  /* Chart geometry in user units. The SVG scales to its container via CSS,
     so these are proportions rather than pixels. */
  var WIDTH = 640;
  var HEIGHT = 200;
  var PAD_TOP = 18;
  var PAD_BOTTOM = 34;
  var PAD_LEFT = 34;
  var PAD_RIGHT = 8;
  var MAX_BARS = 40;

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

  /* "2026-09-26" -> "26 Sep". Short enough to fit without rotating. */
  function shortLabel(label) {
    var match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(label);
    if (match) {
      var months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
      return parseInt(match[3], 10) + " " + months[parseInt(match[2], 10) - 1];
    }
    /* ISO weeks ("2026-W39") and anything else stay as they are. */
    return label;
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
      if (window.console && window.console.warn) {
        window.console.warn("dashboard: could not parse " + attribute, error);
      }
      return null;
    }
  }

  /* A "nice" axis maximum: the smallest round number at or above the peak, so
     the top gridline is a meaningful value rather than the raw maximum. */
  function niceMax(peak) {
    if (peak <= 0) {
      return 1;
    }
    if (peak <= 5) {
      return peak;
    }
    var magnitude = Math.pow(10, Math.floor(Math.log(peak) / Math.LN10));
    var normalised = peak / magnitude;
    var step = normalised <= 1 ? 1
             : normalised <= 2 ? 2
             : normalised <= 5 ? 5
             : 10;
    return step * magnitude;
  }

  function renderBarChart(container) {
    var labels = readJSON(container, "data-labels");
    var values = readJSON(container, "data-values");

    if (!Array.isArray(labels) || !Array.isArray(values) ||
        labels.length === 0) {
      /* Nothing measured. Say so in the chart area rather than leaving a
         blank rectangle that looks like a rendering failure. */
      container.appendChild(text("No data to chart in this range.", {
        x: WIDTH / 2, y: HEIGHT / 2, "text-anchor": "middle",
        "class": "empty-note"
      }));
      return;
    }

    var svg = el("svg", {
      viewBox: "0 0 " + WIDTH + " " + HEIGHT,
      preserveAspectRatio: "xMidYMid meet",
      role: "img"
    });

    /* Too many bars to be readable, so show the most recent ones. The
       service returns chronological order, so the tail is the latest. */
    var start = Math.max(0, values.length - MAX_BARS);
    var shownLabels = labels.slice(start);
    var shownValues = values.slice(start);

    var peak = 0;
    shownValues.forEach(function (value) {
      if (typeof value === "number" && value > peak) {
        peak = value;
      }
    });
    var axisMax = niceMax(peak);

    var plotWidth = WIDTH - PAD_LEFT - PAD_RIGHT;
    var plotHeight = HEIGHT - PAD_TOP - PAD_BOTTOM;
    var slot = plotWidth / shownValues.length;
    var barWidth = Math.max(2, Math.min(slot * 0.68, 46));

    /* Gridlines and y labels, at 0%, 50% and 100% of the axis. */
    [0, 0.5, 1].forEach(function (fraction) {
      var y = PAD_TOP + plotHeight * (1 - fraction);
      svg.appendChild(el("line", {
        x1: PAD_LEFT, x2: WIDTH - PAD_RIGHT, y1: y, y2: y, "class": "grid"
      }));
      svg.appendChild(text(String(Math.round(axisMax * fraction)), {
        x: PAD_LEFT - 6, y: y + 3, "text-anchor": "end", "class": "axis-label"
      }));
    });

    /* Every nth x label, so the axis stays readable at any width. */
    var stride = Math.max(1, Math.ceil(shownLabels.length / 8));

    shownValues.forEach(function (value, index) {
      var count = typeof value === "number" ? value : 0;
      var barHeight = axisMax > 0 ? (count / axisMax) * plotHeight : 0;
      var x = PAD_LEFT + slot * index + (slot - barWidth) / 2;
      var y = PAD_TOP + plotHeight - barHeight;

      if (barHeight > 0) {
        svg.appendChild(el("rect", {
          x: x, y: y, width: barWidth, height: barHeight, "class": "bar"
        }));
        /* The value on the bar, but only when there is room for it. */
        if (slot > 22 && barHeight > 10) {
          svg.appendChild(text(String(count), {
            x: x + barWidth / 2, y: y - 4, "text-anchor": "middle",
            "class": "value-label"
          }));
        }
      }

      if (index % stride === 0) {
        svg.appendChild(text(shortLabel(shownLabels[index]), {
          x: x + barWidth / 2, y: HEIGHT - PAD_BOTTOM + 14,
          "text-anchor": "middle", "class": "axis-label"
        }));
      }
    });

    container.appendChild(svg);
  }

  function init() {
    var charts = document.querySelectorAll('[data-chart="bar"]');
    for (var i = 0; i < charts.length; i += 1) {
      try {
        renderBarChart(charts[i]);
      } catch (error) {
        /* A chart is decoration. If it fails, the page keeps the numbers. */
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

