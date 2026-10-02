(() => {
  function coefficient2(v) {
    if (v === 1) return 1;
    return (v * (5 * v ** 2 - 3) + 3 * (v ** 2 - 1) ** 2 * Math.atanh(v)) / (2 * v ** 4);
  }

  function coefficient3(v) {
    if (v === 1) return 1;
    return (-8 * v ** 5 + 25 * v ** 3 + 15 * (v ** 2 - 1) ** 2 * Math.atanh(v) - 15 * v) / (2 * v ** 5);
  }

  function coefficient4(v) {
    if (v === 1) return 1;
    return -(81 * v ** 5 - 190 * v ** 3 + 15 * (v ** 2 - 7) * (v ** 2 - 1) ** 2 * Math.atanh(v) + 105 * v) / (4 * v ** 6);
  }

  function render() {
    const element = document.getElementById("forgetting-curve");
    if (!element || !window.Plotly) return;

    const dark = document.body.dataset.mdColorScheme === "slate";
    const foreground = dark ? "#e8eaed" : "#1f2937";
    const grid = dark ? "rgba(232, 234, 237, 0.18)" : "rgba(31, 41, 55, 0.18)";
    const colors = dark ? ["#f2f4f7", "#f97066", "#84adff"] : ["#101828", "#d92d20", "#175cd3"];
    const v = Array.from({ length: 501 }, (_, index) => 1 - 0.9 * index / 500);
    const reciprocalV = v.map((value) => 1 / value);
    const values = [coefficient2, coefficient3, coefficient4].map((coefficient) => v.map(coefficient));
    const limits = [
      v.map((value) => 4 * value / 5),
      v.map((value) => 4 * value ** 2 / 7),
      v.map((value) => 8 * value ** 3 / 21)
    ];
    const traces = values.flatMap((y, index) => [
      {
        y,
        x: reciprocalV,
        type: "scatter",
        mode: "lines",
        line: { color: colors[index], width: 3 },
        legendgroup: "l" + (index + 2),
        name: "<i>l</i> = " + (index + 2),
        hoverinfo: "skip"
      },
      {
        y: limits[index],
        x: reciprocalV,
        type: "scatter",
        mode: "lines",
        line: { color: colors[index], dash: "dash", width: 2 },
        legendgroup: "l" + (index + 2),
        showlegend: false,
        hoverinfo: "skip"
      }
    ]);
    traces.push({
      x: reciprocalV,
      y: values[0],
      customdata: v.map((value, index) => [value, values[0][index], values[1][index], values[2][index]]),
      type: "scatter",
      mode: "lines",
      line: { color: "rgba(0, 0, 0, 0)", width: 1 },
      showlegend: false,
      hovertemplate: "1/<i>V</i> = %{x:.6f}<br>&nbsp;&nbsp;&nbsp;<i>V</i> = %{customdata[0]:.6f}<br>&nbsp;&nbsp;<i>A</i><sub>2</sub> = %{customdata[1]:.6f}<br>&nbsp;&nbsp;<i>A</i><sub>3</sub> = %{customdata[2]:.6f}<br>&nbsp;&nbsp;<i>A</i><sub>4</sub> = %{customdata[3]:.6f}<extra></extra>"
    });

    const axis = {
      color: foreground,
      gridcolor: grid,
      linecolor: foreground,
      showline: true,
      zeroline: false
    };
    const layout = {
      autosize: true,
      height: element.clientHeight,
      margin: { l: 72, r: 24, t: 64, b: 64 },
      paper_bgcolor: "rgba(0, 0, 0, 0)",
      plot_bgcolor: "rgba(0, 0, 0, 0)",
      font: { color: foreground, size: 15 },
      hoverlabel: {
        align: "left",
        bgcolor: dark ? "#1f2937" : "#ffffff",
        bordercolor: dark ? "#98a2b3" : "#475467",
        font: { color: foreground, size: 14 }
      },
      hovermode: "x",
      legend: {
        font: { color: foreground, size: 15 },
        groupclick: "togglegroup",
        orientation: "h",
        x: 0.5,
        xanchor: "center",
        y: 1.08,
        yanchor: "bottom"
      },
      xaxis: {
        ...axis,
        range: [1, 10],
        showspikes: false,
        title: { text: "1/<i>V</i>", standoff: 16 }
      },
      yaxis: {
        ...axis,
        range: [-0.03, 1.04],
        title: { text: "<i>A</i><sub>l</sub>", standoff: 14 }
      }
    };

    Plotly.react(element, traces, layout, {
      displaylogo: false,
      responsive: true,
      modeBarButtonsToRemove: ["select2d", "lasso2d"]
    });
  }

  function start() {
    render();
    new MutationObserver(render).observe(document.body, {
      attributes: true,
      attributeFilter: ["data-md-color-scheme"]
    });
  }

  if (document.readyState === "complete") {
    start();
  } else {
    window.addEventListener("load", start, { once: true });
  }
})();
