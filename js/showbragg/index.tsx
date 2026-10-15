/**
 * ShowBragg - the quantem BraggVectors workflow.
 *
 * The top row is a live view in the style of SpyDE's Find Vectors caret: the
 * navigator with a draggable crosshair, the pattern under it with the detected
 * disks, and the detection controls. The kernel re-detects the one pattern under
 * the crosshair on a background thread, newest position first. Template, Bragg
 * vector map, basis and lattice fit sit in collapsed sections below.
 *
 * With the neural method a click on the pattern teaches the detector: on a
 * detection it marks it wrong, again confirms it, again clears it; on empty
 * pattern it marks a missed disk. The kernel owns the marks (`teach_marks`).
 */

import * as React from "react";
import { createRender, useModel, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import MenuItem from "@mui/material/MenuItem";
import Select from "@mui/material/Select";
import Stack from "@mui/material/Stack";
import Switch from "@mui/material/Switch";
import FormControlLabel from "@mui/material/FormControlLabel";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";
import { useTheme, type ThemeColors } from "../theme";
import { preserveRestoredWidgetModelsOnSave } from "../format";
import { useHideStaticFallback } from "../staticFallback";
import {
  IDENTITY_VIEW,
  canvasPoint,
  clamp,
  displayScale,
  drawImage,
  imageToScreen,
  screenToImage,
  usePngBitmap,
  zoomAt,
  type ImageViewport,
  type ViewTransform,
} from "../imageView";

const CANVAS_SIZE = 340;
const HIT_PX = 10;
const CLICK_MOVE_THRESHOLD_PX = 4;
const BASIS_COLORS = ["#ff4d4f", "#40a9ff", "#73d13d"];
const LABEL_CLEARANCE_PX = 26;
const BASIS_LABELS = ["origin", "g1", "g2"] as const;
const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 } as const;
const CMAPS = ["inferno", "magma", "viridis", "plasma", "gray", "turbo"];
const METHODS = [
  { value: "correlation", label: "template correlation" },
  { value: "dog", label: "difference of Gaussians" },
  { value: "neural", label: "neural (DE SpotNet)" },
] as const;

const compactButton = {
  fontSize: 10,
  py: 0.25,
  px: 1,
  minWidth: 0,
  textTransform: "none" as const,
};

type Role = (typeof BASIS_LABELS)[number];
type Marker = { row: number; col: number; color: string; label: string };

function Panel({
  title,
  colors,
  blocked,
  defaultOpen = true,
  children,
}: {
  title: string;
  colors: ThemeColors;
  blocked?: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = React.useState(defaultOpen);

  return (
    <Box sx={{ border: `1px solid ${colors.border}`, borderRadius: 1, mb: `${SPACING.SM}px` }}>
      <Stack
        direction="row"
        justifyContent="space-between"
        alignItems="center"
        onClick={() => setOpen(!open)}
        sx={{ px: `${SPACING.MD}px`, py: `${SPACING.SM}px`, cursor: "pointer", bgcolor: colors.bgAlt }}
      >
        <Typography sx={{ fontSize: 12, fontWeight: 600 }}>{title}</Typography>
        <Typography sx={{ fontSize: 10, color: colors.textMuted }}>{open ? "hide" : "show"}</Typography>
      </Stack>

      {open && (
        <Box sx={{ p: `${SPACING.MD}px` }}>
          {blocked ? (
            <Typography sx={{ fontSize: 11, color: colors.textMuted }}>{blocked}</Typography>
          ) : (
            children
          )}
        </Box>
      )}
    </Box>
  );
}

function NumberField({
  label,
  value,
  onChange,
  step = 1,
  disabled,
  colors,
  hint,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  step?: number;
  disabled?: boolean;
  colors: ThemeColors;
  hint?: string;
}) {
  return (
    <TextField
      label={label}
      title={hint}
      type="number"
      size="small"
      disabled={disabled}
      value={value}
      onChange={(e) => {
        const parsed = Number(e.target.value);
        if (Number.isFinite(parsed)) onChange(parsed);
      }}
      slotProps={{ htmlInput: { step }, inputLabel: { shrink: true } }}
      sx={{
        width: 120,
        "& .MuiInputBase-input": { fontSize: 11, py: 0.5, color: colors.text },
        "& .MuiInputLabel-root": { fontSize: 11, color: colors.textMuted },
      }}
    />
  );
}

function ImageCanvas({
  bytes,
  shape,
  colors,
  overlay,
  markers,
  onMoveMarker,
  onCommitMarker,
  onPick,
  caption,
  size = CANVAS_SIZE,
}: {
  size?: number;
  bytes: DataView | null;
  shape: number[];
  colors: ThemeColors;
  overlay?: (ctx: CanvasRenderingContext2D, viewport: ImageViewport) => void;
  markers?: Marker[];
  onMoveMarker?: (index: number, row: number, col: number) => void;
  onCommitMarker?: (index: number, row: number, col: number) => void;
  onPick?: (row: number, col: number) => void;
  caption?: string;
}) {
  const image = usePngBitmap(bytes);
  const [view, setView] = React.useState<ViewTransform>({ zoom: 1, panX: 0, panY: 0 });
  const baseRef = React.useRef<HTMLCanvasElement>(null);
  const overlayRef = React.useRef<HTMLCanvasElement>(null);
  const containerRef = React.useRef<HTMLDivElement>(null);

  const [height, width] = shape.length === 2 ? shape : [1, 1];
  const viewport: ImageViewport = React.useMemo(
    () => ({ height, width, canvas: size, ...view }),
    [height, width, view, size],
  );

  React.useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const prevent = (e: WheelEvent) => e.preventDefault();
    el.addEventListener("wheel", prevent, { passive: false });
    return () => el.removeEventListener("wheel", prevent);
  }, []);

  React.useLayoutEffect(() => {
    const canvas = baseRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    canvas.width = size;
    canvas.height = size;
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = colors.bg;
    ctx.fillRect(0, 0, size, size);
    if (image) drawImage(ctx, image, viewport);
  }, [image, viewport, colors.bg]);

  React.useLayoutEffect(() => {
    const canvas = overlayRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    canvas.width = size;
    canvas.height = size;
    ctx.clearRect(0, 0, size, size);
    overlay?.(ctx, viewport);

    (markers ?? []).forEach(({ row, col, color, label }) => {
      const [x, y] = imageToScreen(viewport, row, col);
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.font = "bold 11px -apple-system, sans-serif";
      ctx.fillStyle = "#fff";
      ctx.strokeStyle = "rgba(0,0,0,0.85)";
      ctx.lineWidth = 3;
      ctx.strokeText(label, x + 8, y - 6);
      ctx.fillText(label, x + 8, y - 6);
    });
  }, [overlay, markers, viewport]);

  const dragRef = React.useRef<{
    moved: boolean;
    startX: number;
    startY: number;
    panX: number;
    panY: number;
    marker: number;
  } | null>(null);

  const toImage = (e: React.MouseEvent): [number, number] => {
    const canvas = baseRef.current;
    if (!canvas) return [0, 0];
    const [x, y] = canvasPoint(canvas, e);
    return screenToImage(viewport, x, y);
  };

  const hitMarker = (row: number, col: number): number => {
    if (!onMoveMarker) return -1;
    const scale = (size / Math.max(height, width)) * view.zoom;
    const radius = HIT_PX / scale;
    const list = markers ?? [];
    for (let i = list.length - 1; i >= 0; i--) {
      if (Math.hypot(row - list[i].row, col - list[i].col) <= radius) return i;
    }
    return -1;
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    const [row, col] = toImage(e);
    dragRef.current = {
      moved: false,
      startX: e.clientX,
      startY: e.clientY,
      panX: view.panX,
      panY: view.panY,
      marker: hitMarker(row, col),
    };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const drag = dragRef.current;
    if (!drag) return;
    if (Math.hypot(e.clientX - drag.startX, e.clientY - drag.startY) > CLICK_MOVE_THRESHOLD_PX) {
      drag.moved = true;
    }
    if (drag.marker >= 0) {
      const [row, col] = toImage(e);
      onMoveMarker?.(
        drag.marker,
        clamp(row, 0, height - 1),
        clamp(col, 0, width - 1),
      );
      return;
    }
    if (drag.moved) {
      setView((v) => ({
        ...v,
        panX: drag.panX + (e.clientX - drag.startX),
        panY: drag.panY + (e.clientY - drag.startY),
      }));
    }
  };

  const handleMouseUp = (e: React.MouseEvent) => {
    const drag = dragRef.current;
    dragRef.current = null;
    if (!drag) return;
    if (drag.marker >= 0) {
      const [row, col] = toImage(e);
      onCommitMarker?.(
        drag.marker,
        clamp(row, 0, height - 1),
        clamp(col, 0, width - 1),
      );
      return;
    }
    if (drag.moved) return;
    const [row, col] = toImage(e);
    onPick?.(row, col);
  };

  return (
    <Box>
      <Box
        ref={containerRef}
        sx={{
          position: "relative",
          width: size,
          height: size,
          border: `1px solid ${colors.border}`,
          overflow: "hidden",
        }}
      >
        <canvas
          ref={baseRef}
          style={{ position: "absolute", top: 0, left: 0, width: size, height: size, imageRendering: "pixelated" }}
        />
        <canvas
          ref={overlayRef}
          style={{ position: "absolute", top: 0, left: 0, width: size, height: size, pointerEvents: "none" }}
        />
        <canvas
          width={size}
          height={size}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: size,
            height: size,
            opacity: 0,
            cursor: onPick || onMoveMarker ? "crosshair" : "grab",
          }}
          onWheel={(e) => {
            const canvas = baseRef.current;
            if (!canvas) return;
            const [x, y] = canvasPoint(canvas, e);
            setView(zoomAt(viewport, x, y, e.deltaY));
          }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={() => { dragRef.current = null; }}
          onDoubleClick={() => setView({ zoom: 1, panX: 0, panY: 0 })}
        />
      </Box>

      {caption && (
        <Typography sx={{ fontSize: 10, color: colors.textMuted, mt: `${SPACING.XS}px` }}>
          {caption}
        </Typography>
      )}
    </Box>
  );
}


// ---------------------------------------------------------------------------
// The live view, styled after SpyDE's Find Vectors caret: its palette in a dark
// theme and the matching light palette in a light one.
// ---------------------------------------------------------------------------

type Palette = {
  base: string; mantle: string; crust: string; surface0: string; surface1: string;
  overlay0: string; subtext0: string; text: string; accent: string; onAccent: string;
  warn: string; hint: string;
};
const DARK_PALETTE: Palette = {
  base: "#1e1e2e", mantle: "#181825", crust: "#11111b", surface0: "#313244",
  surface1: "#45475a", overlay0: "#6c7086", subtext0: "#a6adc8", text: "#cdd6f4",
  accent: "#89b4fa", onAccent: "#11111b", warn: "#ff9a3c", hint: "#f9e2af",
};
const LIGHT_PALETTE: Palette = {
  base: "#eff1f5", mantle: "#e6e9ef", crust: "#dce0e8", surface0: "#ccd0da",
  surface1: "#bcc0cc", overlay0: "#8c8fa1", subtext0: "#5c5f77", text: "#4c4f69",
  accent: "#1e66f5", onAccent: "#eff1f5", warn: "#fe640b", hint: "#df8e1d",
};
function paletteColors(p: Palette): ThemeColors {
  return {
    bg: p.base, bgAlt: p.mantle, text: p.text, textMuted: p.subtext0,
    border: p.surface0, controlBg: p.crust, accent: p.accent,
  };
}

const SPOT_COLOR = "#ff3030";
const MISSED_COLOR = "#40d070";
const WRONG_COLOR = "#ff9a3c";
const PANEL_WIDTH = 290;
const ROW_GAP = 10;

/** Navigator and pattern sizes that keep the live row on one line in the cell's
 *  width, up to the sizes SpyDE uses. */
function liveSizes(width: number): { navigator: number; pattern: number } {
  const room = Math.max(360, width - PANEL_WIDTH - 2 * ROW_GAP - 2 * 12 - 4);
  const pattern = Math.round(Math.min(380, Math.max(220, room * 0.56)));
  const navigator = Math.round(Math.min(320, Math.max(140, room - pattern)));
  return { navigator, pattern };
}

type ModelEntry = { id: string; label: string; subtitle?: string; icon?: string; taught?: boolean };
type TeachState = {
  wrong?: number; missed?: number; confirmed?: number; busy?: boolean; taught?: boolean;
  learned?: number; total?: number; original_f1?: number | null; original_f1_base?: number | null;
  saved?: string; summary?: string;
};

/** The navigator: drag the crosshair, or step it with the arrow keys. */
function NavigatorCanvas({
  bytes, shape, position, onMove, palette, size,
}: {
  size: number;
  bytes: DataView | null;
  shape: number[];
  position: number[];
  onMove: (row: number, col: number) => void;
  palette: Palette;
}) {
  const image = usePngBitmap(bytes);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const dragging = React.useRef(false);
  const pending = React.useRef<[number, number] | null>(null);
  const frame = React.useRef<number | null>(null);
  const [rows, cols] = shape.length === 2 ? shape : [1, 1];
  const viewport: ImageViewport = React.useMemo(
    () => ({ height: rows, width: cols, canvas: size, ...IDENTITY_VIEW }),
    [rows, cols, size],
  );

  React.useLayoutEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    canvas.width = size;
    canvas.height = size;
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = palette.crust;
    ctx.fillRect(0, 0, size, size);
    if (image) drawImage(ctx, image, viewport);
    const [x, y] = imageToScreen(viewport, position[0] + 0.5, position[1] + 0.5);
    const cell = displayScale(viewport);
    ctx.strokeStyle = "#ff3030";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, size);
    ctx.moveTo(0, y);
    ctx.lineTo(size, y);
    ctx.stroke();
    ctx.strokeRect(x - cell / 2, y - cell / 2, cell, cell);
  }, [image, viewport, position, palette]);

  // A fast drag only ever sends the newest cell, at most once per animation frame.
  const flush = () => {
    frame.current = null;
    if (pending.current) {
      const [row, col] = pending.current;
      pending.current = null;
      if (row !== position[0] || col !== position[1]) onMove(row, col);
    }
  };
  const moveTo = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const [x, y] = canvasPoint(canvas, e);
    const [row, col] = screenToImage(viewport, x, y);
    pending.current = [clamp(Math.floor(row), 0, rows - 1), clamp(Math.floor(col), 0, cols - 1)];
    if (frame.current === null) frame.current = requestAnimationFrame(flush);
  };
  const onKey = (e: React.KeyboardEvent) => {
    const step: Record<string, [number, number]> = {
      ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1],
    };
    const delta = step[e.key];
    if (!delta) return;
    e.preventDefault();
    onMove(
      clamp(position[0] + delta[0], 0, rows - 1),
      clamp(position[1] + delta[1], 0, cols - 1),
    );
  };

  return (
    <canvas
      ref={canvasRef}
      tabIndex={0}
      aria-label="Navigator: drag the crosshair, or use the arrow keys"
      data-testid="showbragg-navigator"
      style={{
        width: size, height: size, display: "block", cursor: "crosshair",
        border: `1px solid ${palette.surface0}`, borderRadius: 4, imageRendering: "pixelated",
        outline: "none",
      }}
      onMouseDown={(e) => { dragging.current = true; (e.target as HTMLCanvasElement).focus(); moveTo(e); }}
      onMouseMove={(e) => { if (dragging.current) moveTo(e); }}
      onMouseUp={() => { dragging.current = false; }}
      onMouseLeave={() => { dragging.current = false; }}
      onKeyDown={onKey}
    />
  );
}

/** Label, slider and an editable number, on one line. */
function SliderRow({
  label, value, min, max, step, onChange, palette, title, slider = true,
}: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void; palette: Palette; title?: string; slider?: boolean;
}) {
  const [draft, setDraft] = React.useState<string | null>(null);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }} title={title}>
      <span style={{ fontSize: 10.5, color: palette.subtext0, width: 74, flex: "0 0 auto", whiteSpace: "nowrap" }}>
        {label}
      </span>
      {slider ? (
        <input
          type="range" min={min} max={max} step={step} value={clamp(value, min, max)}
          aria-label={label}
          style={{ flex: 1, minWidth: 40, accentColor: palette.accent }}
          onChange={(e) => onChange(Number(e.target.value))}
        />
      ) : (
        <span style={{ flex: 1 }} />
      )}
      <input
        type="number" step={step} min={min}
        aria-label={`${label} value`}
        value={draft ?? String(value)}
        style={{
          width: 52, background: palette.crust, color: palette.text,
          border: `1px solid ${palette.surface0}`, borderRadius: 4, padding: "2px 4px", fontSize: 11,
        }}
        onChange={(e) => {
          setDraft(e.target.value);
          const n = Number(e.target.value);
          if (e.target.value !== "" && Number.isFinite(n)) onChange(n);
        }}
        onBlur={() => setDraft(null)}
      />
    </div>
  );
}

/** SpyDE's small on/off switch. */
function PaletteToggle({
  checked, onChange, label, palette, title, disabled = false,
}: {
  checked: boolean; onChange: (v: boolean) => void; label: string; palette: Palette;
  title?: string; disabled?: boolean;
}) {
  return (
    <button
      type="button" role="switch" aria-checked={checked} title={title} disabled={disabled}
      onClick={() => onChange(!checked)}
      style={{
        display: "flex", alignItems: "center", gap: 5, background: "none", border: "none", padding: 0,
        color: palette.subtext0, fontSize: 10.5, cursor: disabled ? "default" : "pointer",
        whiteSpace: "nowrap", opacity: disabled ? 0.45 : 1,
      }}
    >
      <span
        style={{
          position: "relative", width: 22, height: 12, borderRadius: 6, boxSizing: "border-box",
          background: checked ? palette.accent : palette.surface0,
          border: `1px solid ${checked ? palette.accent : palette.surface1}`, flex: "0 0 auto",
        }}
      >
        <span
          style={{
            position: "absolute", top: 1, left: checked ? 11 : 1, width: 8, height: 8, borderRadius: 4,
            background: checked ? palette.onAccent : palette.subtext0,
          }}
        />
      </span>
      {label}
    </button>
  );
}

function PaletteSelect({
  value, options, onChange, palette, label,
}: {
  value: string; options: { value: string; label: string }[]; onChange: (v: string) => void;
  palette: Palette; label: string;
}) {
  return (
    <select
      aria-label={label} value={value} onChange={(e) => onChange(e.target.value)}
      style={{
        flex: 1, minWidth: 0, background: palette.crust, color: palette.text,
        border: `1px solid ${palette.surface0}`, borderRadius: 4, padding: "3px 5px", fontSize: 11,
      }}
    >
      {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
    </select>
  );
}

function ModelIcon({ entry, size, palette }: { entry?: ModelEntry; size: number; palette: Palette }) {
  const box: React.CSSProperties = {
    width: size, height: size, borderRadius: 4, flex: "0 0 auto",
    border: `1px solid ${entry?.taught ? palette.hint : palette.surface1}`,
    background: palette.surface0, display: "flex", alignItems: "center", justifyContent: "center",
    fontSize: Math.round(size * 0.36), color: palette.subtext0, overflow: "hidden",
  };
  if (entry?.icon) return <img src={entry.icon} alt="" style={{ ...box, objectFit: "cover" }} draggable={false} />;
  const initials = (entry?.label ?? "?").split(/[\s_\-:]+/).filter(Boolean).slice(0, 2).map((w) => w[0]).join("");
  return <span style={box}>{initials.toUpperCase()}</span>;
}

/** The detector menu: each row has the model's icon (the disk it most likes to see). */
function ModelMenu({
  models, value, onChange, palette,
}: {
  models: ModelEntry[]; value: string; onChange: (id: string) => void; palette: Palette;
}) {
  const [open, setOpen] = React.useState(false);
  const current = models.find((m) => m.id === value);
  return (
    <div style={{ position: "relative", flex: 1, minWidth: 0 }}>
      <button
        type="button" aria-haspopup="listbox" aria-expanded={open} data-testid="showbragg-model"
        onClick={() => setOpen(!open)}
        style={{
          width: "100%", display: "flex", alignItems: "center", gap: 6, background: palette.crust,
          color: palette.text, border: `1px solid ${palette.surface0}`, borderRadius: 4,
          padding: "2px 5px", fontSize: 11, cursor: "pointer", textAlign: "left",
        }}
      >
        <ModelIcon entry={current} size={18} palette={palette} />
        <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {current?.label ?? value}
        </span>
        <span style={{ fontSize: 9, color: palette.overlay0 }}>▾</span>
      </button>
      {open && (
        <div
          role="listbox"
          style={{
            position: "absolute", zIndex: 20, top: "calc(100% + 2px)", left: 0, right: 0,
            background: palette.base, border: `1px solid ${palette.surface1}`, borderRadius: 6,
            boxShadow: "0 8px 24px rgba(0,0,0,0.35)", padding: 3,
          }}
        >
          {models.map((m) => (
            <button
              key={m.id} type="button" role="option" aria-selected={m.id === value}
              onClick={() => { onChange(m.id); setOpen(false); }}
              style={{
                width: "100%", display: "flex", alignItems: "center", gap: 7, border: "none",
                background: m.id === value ? palette.surface0 : "transparent", color: palette.text,
                padding: "3px 4px", borderRadius: 4, cursor: "pointer", textAlign: "left",
              }}
            >
              <ModelIcon entry={m} size={28} palette={palette} />
              <span style={{ display: "flex", flexDirection: "column", minWidth: 0 }}>
                <span style={{ fontSize: 11, fontWeight: m.id === value ? 600 : 400 }}>{m.label}</span>
                <span style={{ fontSize: 9.5, color: palette.overlay0 }}>{m.subtitle}</span>
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function SmallButton({
  onClick, disabled = false, palette, children, title,
}: {
  onClick: () => void; disabled?: boolean; palette: Palette; children: React.ReactNode; title?: string;
}) {
  return (
    <button
      type="button" onClick={onClick} disabled={disabled} title={title}
      style={{
        padding: "1px 7px", fontSize: 11, background: "transparent", color: palette.text,
        border: `1px solid ${palette.surface1}`, borderRadius: 4,
        cursor: disabled ? "default" : "pointer", opacity: disabled ? 0.4 : 1,
      }}
    >
      {children}
    </button>
  );
}

function ShowBragg() {
  const model = useModel();
  const rootRef = React.useRef<HTMLDivElement>(null);
  const { themeInfo } = useTheme();
  const palette = themeInfo.theme === "dark" ? DARK_PALETTE : LIGHT_PALETTE;
  const colors = paletteColors(palette);

  React.useEffect(() => preserveRestoredWidgetModelsOnSave(model), [model]);
  useHideStaticFallback(model, rootRef);

  const [title] = useModelState<string>("title");
  const [showTitle] = useModelState<boolean>("show_title");
  const [showControls] = useModelState<boolean>("show_controls");
  const [controlsCollapsed] = useModelState<boolean>("controls_collapsed");
  const controlsVisible = showControls && !controlsCollapsed;
  const [scanShape] = useModelState<number[]>("scan_shape");
  const [qShape] = useModelState<number[]>("q_shape");
  const [status] = useModelState<string>("status");
  const [cmap, setCmap] = useModelState<string>("cmap");

  const [templateSource, setTemplateSource] = useModelState<string>("template_source");
  const [templateRadius, setTemplateRadius] = useModelState<number>("template_radius");
  const [templateEdge, setTemplateEdge] = useModelState<number>("template_edge");
  const [templateSubtractMean, setTemplateSubtractMean] = useModelState<boolean>("template_subtract_mean");
  const [templatePng] = useModelState<DataView>("template_png");
  const [templateShape] = useModelState<number[]>("template_shape");
  const [hasProbe] = useModelState<boolean>("has_probe");

  const [probePosition, setProbePosition] = useModelState<number[]>("probe_position");
  const [navigatorPng] = useModelState<DataView>("navigator_png");
  const [navigatorKind, setNavigatorKind] = useModelState<string>("navigator_kind");
  const [patternPng] = useModelState<DataView>("pattern_png");
  const [patternLog, setPatternLog] = useModelState<boolean>("pattern_log");
  const [showResponse, setShowResponse] = useModelState<boolean>("show_response");
  const [spots] = useModelState<number[][]>("spots");
  const [spotsPosition] = useModelState<number[]>("spots_position");
  const [spotRadiusShown] = useModelState<number>("spot_display_radius");
  const [liveStatus] = useModelState<string>("live_status");
  const [models] = useModelState<ModelEntry[]>("models");
  const [neuralModel, setNeuralModel] = useModelState<string>("neural_model");
  const [teachMarks] = useModelState<Record<string, number[][]>>("teach_marks");
  const [teachState] = useModelState<TeachState>("teach_state");
  const [neighborRefine, setNeighborRefine] = useModelState<boolean>("neighbor_refine");
  const [extraOpen, setExtraOpen] = React.useState(false);
  const [savePath, setSavePath] = React.useState<string | null>(null);

  const [detectionMethod, setDetectionMethod] = useModelState<string>("detection_method");
  const [threshold, setThreshold] = useModelState<number>("threshold");
  const [dogSigma1, setDogSigma1] = useModelState<number>("dog_sigma1");
  const [dogSigma2, setDogSigma2] = useModelState<number>("dog_sigma2");
  const [spotRadius, setSpotRadius] = useModelState<number>("spot_radius");
  const [maskBeamstop, setMaskBeamstop] = useModelState<boolean>("mask_beamstop");
  const [refineCenters, setRefineCenters] = useModelState<boolean>("refine_centers");
  const [friedelPartner, setFriedelPartner] = useModelState<boolean>("friedel_partner");
  const [minAbsIntensity, setMinAbsIntensity] = useModelState<number>("min_abs_intensity");
  const [minSpacing, setMinSpacing] = useModelState<number>("min_spacing");
  const [edgeBoundary, setEdgeBoundary] = useModelState<number>("edge_boundary");
  const [subpixel, setSubpixel] = useModelState<string>("subpixel");
  const [upsampleFactor, setUpsampleFactor] = useModelState<number>("upsample_factor");
  const [maxNumPeaks, setMaxNumPeaks] = useModelState<number>("max_num_peaks");
  const [detectionState] = useModelState<string>("detection_state");

  const [bvmSampling, setBvmSampling] = useModelState<number>("bvm_sampling");
  const [bvmPng] = useModelState<DataView>("bvm_png");
  const [numCandidates, setNumCandidates] = useModelState<number>("num_candidates");
  const [candidateMinSpacing, setCandidateMinSpacing] = useModelState<number>("candidate_min_spacing");
  const [candidateMinAbsIntensity, setCandidateMinAbsIntensity] = useModelState<number>("candidate_min_abs_intensity");
  const [candidates] = useModelState<number[][]>("candidates");
  const [originIndex, setOriginIndex] = useModelState<number>("origin_index");
  const [g1Index, setG1Index] = useModelState<number>("g1_index");
  const [g2Index, setG2Index] = useModelState<number>("g2_index");
  const [originRc, setOriginRc] = useModelState<number[]>("origin_rc");
  const [g1Rc, setG1Rc] = useModelState<number[]>("g1_rc");
  const [g2Rc, setG2Rc] = useModelState<number[]>("g2_rc");

  const [minNumPeaks, setMinNumPeaks] = useModelState<number>("min_num_peaks");
  const [maxPeakShift, setMaxPeakShift] = useModelState<number>("max_peak_shift");
  const [fitState] = useModelState<string>("fit_state");
  const [maskWeightPng] = useModelState<DataView>("mask_weight_png");
  const [fitErrorPng] = useModelState<DataView>("fit_error_png");

  const send = React.useCallback(
    (content: Record<string, unknown>) => model.send(content),
    [model],
  );

  const detecting = detectionState === "running" || detectionState === "preview";
  const fitting = fitState === "running";
  const busy = detecting || fitting;
  const hasPeaks = candidates.length > 0 || bvmPng?.byteLength > 0;
  const usesTemplate = detectionMethod === "correlation";

  const drawCandidates = React.useCallback(
    (ctx: CanvasRenderingContext2D, viewport: ImageViewport) => {
      const placed: Array<[number, number]> = [];
      const byBrightness = candidates
        .map((c, i) => ({ row: c[0], col: c[1], intensity: c[2] ?? 0, index: i }))
        .sort((a, b) => b.intensity - a.intensity);

      ctx.font = "10px -apple-system, sans-serif";
      for (const { row, col, index } of byBrightness) {
        const [x, y] = imageToScreen(viewport, row, col);
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.strokeStyle = "rgba(255,255,255,0.8)";
        ctx.lineWidth = 1;
        ctx.stroke();

        const clear = placed.every(
          ([px, py]) => Math.hypot(x - px, y - py) >= LABEL_CLEARANCE_PX,
        );
        if (!clear) continue;
        placed.push([x, y]);

        ctx.fillStyle = "#fff";
        ctx.strokeStyle = "rgba(0,0,0,0.85)";
        ctx.lineWidth = 2.5;
        ctx.strokeText(String(index), x + 4, y - 4);
        ctx.fillText(String(index), x + 4, y - 4);
      }
    },
    [candidates],
  );

  const [dragPreview, setDragPreview] = React.useState<[number, number, number] | null>(null);

  const basisMarkers: Marker[] = React.useMemo(() => {
    if (originRc.length !== 2) return [];
    const tips = [originRc, g1Rc, g2Rc];
    return tips.flatMap((rc, i) => {
      if (rc.length !== 2) return [];
      if (dragPreview && dragPreview[0] === i) {
        return [{ row: dragPreview[1], col: dragPreview[2], color: BASIS_COLORS[i], label: BASIS_LABELS[i] }];
      }
      const row = i === 0 ? rc[0] : originRc[0] + rc[0];
      const col = i === 0 ? rc[1] : originRc[1] + rc[1];
      return [{ row, col, color: BASIS_COLORS[i], label: BASIS_LABELS[i] }];
    });
  }, [originRc, g1Rc, g2Rc, dragPreview]);

  const setRoleVector = (index: number, row: number, col: number) => {
    setDragPreview(null);
    if (index === 0) {
      setOriginIndex(-1);
      setOriginRc([row, col]);
    } else if (index === 1) {
      setG1Index(-1);
      setG1Rc([row - originRc[0], col - originRc[1]]);
    } else {
      setG2Index(-1);
      setG2Rc([row - originRc[0], col - originRc[1]]);
    }
  };

  const [activeRole, setActiveRole] = React.useState<Role>("origin");

  const pickCandidate = (row: number, col: number) => {
    if (!candidates.length) return;
    let nearest = 0;
    let best = Infinity;
    candidates.forEach(([cr, cc], i) => {
      const d = Math.hypot(row - cr, col - cc);
      if (d < best) {
        best = d;
        nearest = i;
      }
    });
    if (activeRole === "origin") setOriginIndex(nearest);
    else if (activeRole === "g1") setG1Index(nearest);
    else setG2Index(nearest);
  };

  const vectorText = (rc: number[]) =>
    rc.length === 2 ? `(${rc[0].toFixed(2)}, ${rc[1].toFixed(2)})` : "not set";

  const isNeural = detectionMethod === "neural";
  const isDog = detectionMethod === "dog";
  const marksHere: number[][] = teachMarks[`${probePosition[0]},${probePosition[1]}`] ?? [];
  const spotsShown = spotsPosition.length === 2
    && spotsPosition[0] === probePosition[0] && spotsPosition[1] === probePosition[1];

  const drawSpots = React.useCallback(
    (ctx: CanvasRenderingContext2D, viewport: ImageViewport) => {
      const scale = displayScale(viewport) * viewport.zoom;
      const r = Math.max(3, spotRadiusShown * scale);
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = SPOT_COLOR;
      for (const [row, col] of spots) {
        const [x, y] = imageToScreen(viewport, row, col);
        ctx.beginPath();
        ctx.arc(x, y, r, 0, 2 * Math.PI);
        ctx.stroke();
      }
      for (const [row, col, label] of marksHere) {
        const [x, y] = imageToScreen(viewport, row, col);
        if (label === 0) {
          const arm = 0.7 * r;
          ctx.strokeStyle = WRONG_COLOR;
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          ctx.moveTo(x - arm, y - arm);
          ctx.lineTo(x + arm, y + arm);
          ctx.moveTo(x - arm, y + arm);
          ctx.lineTo(x + arm, y - arm);
          ctx.stroke();
        } else {
          ctx.strokeStyle = MISSED_COLOR;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(x, y, r + 2.5, 0, 2 * Math.PI);
          ctx.stroke();
        }
      }
    },
    [spots, marksHere, spotRadiusShown],
  );

  const [rowWidth, setRowWidth] = React.useState(1000);
  React.useEffect(() => {
    const element = rootRef.current;
    if (!element || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(([entry]) => setRowWidth(entry.contentRect.width));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  const sizes = liveSizes(rowWidth);

  const panelBox: React.CSSProperties = {
    width: PANEL_WIDTH, flex: "0 0 auto", background: palette.base,
    border: `1px solid ${palette.surface0}`, borderRadius: 8, padding: 8,
    display: "flex", flexDirection: "column", gap: 6, color: palette.text,
    boxSizing: "border-box", alignSelf: "stretch",
  };
  const row: React.CSSProperties = { display: "flex", alignItems: "center", gap: 6, minWidth: 0 };
  const rowLabel: React.CSSProperties = { fontSize: 10.5, color: palette.subtext0, width: 40, flex: "0 0 auto" };
  const paneHead: React.CSSProperties = {
    display: "flex", alignItems: "center", justifyContent: "space-between", gap: 6,
    fontSize: 10.5, color: palette.subtext0, height: 20,
  };
  const counts = teachState ?? {};
  const markCount = (counts.wrong ?? 0) + (counts.missed ?? 0) + (counts.confirmed ?? 0);
  const defaultSaveName = `${(title || "dataset").replace(/[^A-Za-z0-9_-]+/g, "_")}_spotnet_taught.pt`;

  return (
    <Box
      ref={rootRef}
      sx={{ p: `${SPACING.MD}px`, bgcolor: palette.mantle, color: palette.text, width: "100%", boxSizing: "border-box", borderRadius: 1 }}
    >
      {showTitle && (
        <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.SM}px` }}>
          <Typography sx={{ fontSize: 12.5, fontWeight: 600, color: palette.text }}>{title || "Show Bragg"}</Typography>
          <Stack direction="row" spacing={1} alignItems="center">
            <Typography sx={{ fontSize: 10, color: palette.overlay0 }}>
              scan {scanShape.join(" × ")} · detector {qShape.join(" × ")}
            </Typography>
            <PaletteSelect
              label="colormap" value={cmap} palette={palette} onChange={setCmap}
              options={CMAPS.map((c) => ({ value: c, label: c }))}
            />
          </Stack>
        </Stack>
      )}

      {status && (
        <Typography sx={{ fontSize: 11, color: palette.warn, mb: `${SPACING.SM}px` }}>{status}</Typography>
      )}

      <div style={{ display: "flex", gap: ROW_GAP, alignItems: "flex-start", flexWrap: "wrap" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <div style={paneHead}>
            <span>Navigator · ({probePosition[0]}, {probePosition[1]})</span>
            <div style={{ width: 80, display: "flex" }}>
              <PaletteSelect
                label="navigator image" value={navigatorKind} palette={palette} onChange={setNavigatorKind}
                options={[{ value: "sum", label: "sum" }, { value: "adf", label: "ADF" }]}
              />
            </div>
          </div>
          <NavigatorCanvas
            size={sizes.navigator}
            bytes={navigatorPng} shape={scanShape} position={probePosition}
            palette={palette} onMove={(r, c) => setProbePosition([r, c])}
          />
          <span style={{ fontSize: 10, color: palette.overlay0, fontStyle: "italic" }}>
            drag the crosshair · arrow keys step it
          </span>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <div style={paneHead}>
            <span>{showResponse ? "Response" : "Pattern"} · {spotsShown ? spots.length : "…"} spots</span>
            <PaletteToggle checked={Boolean(patternLog)} onChange={setPatternLog} label="log" palette={palette} />
          </div>
          <ImageCanvas
            bytes={patternPng}
            shape={qShape}
            colors={colors}
            size={sizes.pattern}
            overlay={drawSpots}
            onPick={isNeural ? (r, c) => send({ type: "mark", row: r, col: c }) : undefined}
          />
          <span style={{ fontSize: 10, color: palette.overlay0, fontStyle: "italic" }}>
            {isNeural
              ? "click a circle that is wrong (again: right) or a disk it missed · scroll to zoom"
              : "scroll to zoom · drag to pan · double-click to reset"}
          </span>
        </div>

        {controlsVisible && (
          <div style={panelBox} data-testid="showbragg-detect-panel">
            <div style={{ fontSize: 11, fontWeight: 600 }}>Find disks</div>
            <div style={row}>
              <span style={rowLabel}>Method</span>
              <PaletteSelect
                label="detection method" value={detectionMethod} palette={palette} onChange={setDetectionMethod}
                options={METHODS.map((m) => ({ value: m.value, label: m.label }))}
              />
              <PaletteToggle
                checked={Boolean(showResponse)} onChange={setShowResponse} palette={palette}
                label={isNeural ? "Heatmap" : isDog ? "DoG" : "Corr."}
                title="Show what the method thresholds instead of the pattern"
              />
            </div>
            {isNeural && models.length > 0 && (
              <div style={row}>
                <span style={rowLabel}>Model</span>
                <ModelMenu models={models} value={neuralModel} onChange={setNeuralModel} palette={palette} />
              </div>
            )}

            {isNeural && (
              <SliderRow label="Spot size (px)" value={spotRadius} min={0} max={30} step={0.5}
                onChange={setSpotRadius} palette={palette} title="disk radius; 0 estimates it from the data" />
            )}
            {isDog && (
              <>
                <SliderRow label="DoG σ₁ (spot)" value={dogSigma1} min={0.4} max={3} step={0.1}
                  onChange={setDogSigma1} palette={palette} />
                <SliderRow label="DoG σ₂ (bg)" value={dogSigma2} min={1} max={6} step={0.1}
                  onChange={setDogSigma2} palette={palette} />
              </>
            )}
            {usesTemplate ? (
              <SliderRow label="Min intensity" value={minAbsIntensity} min={0} max={1} step={0.01}
                slider={false} onChange={setMinAbsIntensity} palette={palette}
                title="drop correlation peaks below this absolute intensity" />
            ) : (
              <SliderRow
                label={isDog ? "Threshold (SNR)" : "Threshold"} value={threshold}
                min={0} max={isDog ? 30 : 1} step={isDog ? 0.5 : 0.01}
                onChange={setThreshold} palette={palette}
                title={isDog ? "band-pass signal-to-noise ratio" : "network confidence, 0 to 1"}
              />
            )}
            <SliderRow label="Min spacing" value={minSpacing} min={0} max={30} step={0.5}
              onChange={setMinSpacing} palette={palette} title="pixels between kept peaks" />

            <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 12px" }}>
              {isNeural && (
                <>
                  <PaletteToggle checked={Boolean(refineCenters)} onChange={setRefineCenters} palette={palette}
                    label="Refine centers" title="re-place each disk on the raw pattern with DE CenterNet (spots of 5 px radius or more)" />
                  <PaletteToggle checked={Boolean(friedelPartner) && Boolean(refineCenters)} onChange={setFriedelPartner}
                    palette={palette} disabled={!refineCenters} label="Friedel partner"
                    title="refine again with each disk's Friedel mirror, using DE FriedelNet (spots of 6-16 px radius)" />
                </>
              )}
              <PaletteToggle checked={Boolean(neighborRefine)} onChange={setNeighborRefine} palette={palette}
                label="Neighbor refine" title="drop detections the neighboring scan positions do not support (full scan only)" />
              {!usesTemplate && (
                <PaletteToggle checked={Boolean(maskBeamstop)} onChange={setMaskBeamstop} palette={palette}
                  label="Beam stop" title="find the beam stop in the mean pattern and ignore detections inside it" />
              )}
            </div>

            <div style={{ borderTop: `1px solid ${palette.surface0}`, paddingTop: 4, display: "flex", flexDirection: "column", gap: 6 }}>
              <button type="button" onClick={() => setExtraOpen(!extraOpen)} aria-expanded={extraOpen}
                style={{ display: "flex", alignItems: "center", gap: 4, background: "none", border: "none", color: palette.subtext0, cursor: "pointer", fontSize: 10.5, padding: 0, alignSelf: "flex-start" }}>
                <span style={{ display: "inline-block", width: 8, transform: extraOpen ? "rotate(90deg)" : "none" }}>›</span>
                Extra
              </button>
              {extraOpen && (
                <>
                  <SliderRow label="Max peaks" value={maxNumPeaks} min={1} max={2000} step={1} slider={false}
                    onChange={setMaxNumPeaks} palette={palette} />
                  {!isNeural && (
                    <>
                      <SliderRow label="Edge" value={edgeBoundary} min={0} max={20} step={1}
                        onChange={setEdgeBoundary} palette={palette} title="border width in which peaks are ignored" />
                      <div style={row}>
                        <span style={{ ...rowLabel, width: 74 }}>Subpixel</span>
                        <PaletteSelect label="subpixel" value={subpixel} palette={palette} onChange={setSubpixel}
                          options={[
                            { value: "none", label: "none" },
                            { value: "parabolic", label: "parabolic" },
                            ...(usesTemplate ? [{ value: "upsample", label: "upsample" }] : []),
                          ]} />
                      </div>
                    </>
                  )}
                  {usesTemplate && (
                    <SliderRow label="Upsample" value={upsampleFactor} min={1} max={64} step={1}
                      onChange={setUpsampleFactor} palette={palette} />
                  )}
                </>
              )}
            </div>

            {isNeural && (
              <div style={{ borderTop: `1px solid ${palette.surface0}`, paddingTop: 6, display: "flex", flexDirection: "column", gap: 4, fontSize: 11 }}
                data-testid="showbragg-teach">
                <span>
                  {markCount === 0
                    ? "Teach: click a wrong circle or a missed disk"
                    : `Teach: ${counts.wrong ?? 0} wrong · ${counts.missed ?? 0} missed · ${counts.confirmed ?? 0} right${counts.busy ? " — teaching…" : ""}`}
                </span>
                <div style={{ display: "flex", gap: 4 }}>
                  <SmallButton palette={palette} disabled={markCount === 0 || Boolean(counts.busy)}
                    onClick={() => send({ type: "teach" })}>Teach</SmallButton>
                  <SmallButton palette={palette} disabled={markCount === 0 && !counts.taught}
                    onClick={() => send({ type: "revert_teaching" })}>Revert</SmallButton>
                  <SmallButton palette={palette} disabled={!counts.taught}
                    title={counts.taught ? "Save the taught detector as a .pt for model=" : "Teach first"}
                    onClick={() => setSavePath(defaultSaveName)}>Save</SmallButton>
                </div>
                {savePath !== null && (
                  <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
                    <input value={savePath} autoFocus aria-label="file to save the taught detector to"
                      onChange={(e) => setSavePath(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") { send({ type: "save_model", path: savePath }); setSavePath(null); }
                        if (e.key === "Escape") setSavePath(null);
                      }}
                      style={{ flex: 1, minWidth: 0, background: palette.crust, color: palette.text, border: `1px solid ${palette.surface0}`, borderRadius: 4, padding: "2px 4px", fontSize: 11 }} />
                    <SmallButton palette={palette} onClick={() => { send({ type: "save_model", path: savePath }); setSavePath(null); }}>✓</SmallButton>
                    <SmallButton palette={palette} onClick={() => setSavePath(null)}>✕</SmallButton>
                  </div>
                )}
                {counts.taught && counts.summary && (
                  <span style={{ fontSize: 10, color: palette.subtext0 }} data-testid="showbragg-teach-summary">
                    {counts.summary}
                  </span>
                )}
                {counts.taught && typeof counts.original_f1 === "number" && (
                  <span style={{ fontSize: 10, color: counts.original_f1 < 0.7 ? palette.warn : palette.subtext0 }}
                    title="F1 on the synthetic patterns the detector was trained on: how far teaching pulled it toward this dataset">
                    {counts.original_f1 < 0.7 ? "Now specific to this dataset" : "Using the taught detector"}
                    {` · general F1 ${counts.original_f1.toFixed(2)}`}
                    {typeof counts.original_f1_base === "number" ? ` (was ${counts.original_f1_base.toFixed(2)})` : ""}
                  </span>
                )}
              </div>
            )}

            <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <button type="button" disabled={busy} onClick={() => send({ type: "run_detect" })}
                style={{
                  background: busy ? palette.surface1 : palette.accent, color: busy ? palette.subtext0 : palette.onAccent,
                  border: "none", borderRadius: 5, padding: "5px 10px", fontSize: 11.5, fontWeight: 600,
                  cursor: busy ? "progress" : "pointer",
                }}>
                {detectionState === "running" ? "Detecting…" : "Run on whole scan"}
              </button>
            </div>
            <div style={{ fontSize: 10, color: palette.subtext0, borderTop: `1px solid ${palette.surface0}`, paddingTop: 4 }}
              data-testid="showbragg-status">
              {liveStatus || "Drag the crosshair to preview the found disks."}
            </div>
          </div>
        )}
      </div>

      <div style={{ marginTop: SPACING.MD }}>
      <Panel
        title="Template (correlation method)"
        defaultOpen={false}
        colors={colors}
      >
        <Stack direction="row" spacing={2} alignItems="flex-start" flexWrap="wrap">
          {controlsVisible && (
          <Stack spacing={1.5} sx={{ minWidth: 200 }}>
            <Select
              size="small"
              value={templateSource}
              onChange={(e) => setTemplateSource(e.target.value)}
              sx={{ fontSize: 11, "& .MuiSelect-select": { py: 0.5, color: colors.text } }}
            >
              <MenuItem sx={{ fontSize: 11 }} value="synthetic">synthetic disk</MenuItem>
              <MenuItem sx={{ fontSize: 11 }} value="data">mean of data</MenuItem>
              <MenuItem sx={{ fontSize: 11 }} value="probe" disabled={!hasProbe}>measured probe</MenuItem>
            </Select>

            <NumberField
              label="radius"
              hint="disk radius in pixels; 0 estimates it"
              value={templateRadius}
              step={0.5}
              disabled={templateSource !== "synthetic"}
              onChange={setTemplateRadius}
              colors={colors}
            />
            <NumberField
              label="edge"
              hint="tanh edge falloff width"
              value={templateEdge}
              step={0.25}
              disabled={templateSource !== "synthetic"}
              onChange={setTemplateEdge}
              colors={colors}
            />
            <FormControlLabel
              control={
                <Switch
                  size="small"
                  checked={Boolean(templateSubtractMean)}
                  onChange={(e) => setTemplateSubtractMean(e.target.checked)}
                />
              }
              label="Subtract Mean"
              slotProps={{ typography: { sx: { fontSize: 11, color: colors.text } } }}
              sx={{ alignSelf: "flex-start", ml: 0 }}
            />
          </Stack>
          )}

          <ImageCanvas
            bytes={templatePng}
            shape={templateShape.length === 2 && templateShape[0] > 0 ? templateShape : qShape}
            colors={colors}
            caption="correlation template, cropped to the disk"
          />
        </Stack>
      </Panel>

      <Panel
        title="Bragg vector map"
        defaultOpen={false}
        colors={colors}
        blocked={hasPeaks ? undefined : "Run Detection to accumulate the Bragg vector map."}
      >
        <Stack direction="row" spacing={2} alignItems="flex-start" flexWrap="wrap">
          {controlsVisible && (
          <Stack spacing={1.5} sx={{ minWidth: 200 }}>
            <NumberField label="bvm_sampling" value={bvmSampling} step={0.1} onChange={setBvmSampling} colors={colors} />
            <Button
              size="small"
              sx={{ ...compactButton, color: colors.accent, alignSelf: "flex-start" }}
              disabled={busy}
              onClick={() => send({ type: "compute_bvm" })}
            >
              Recompute Map
            </Button>
          </Stack>
          )}

          <ImageCanvas
            bytes={bvmPng}
            shape={qShape}
            colors={colors}
            overlay={drawCandidates}
            caption={`${candidates.length} candidate peaks${numCandidates > 0 ? "" : " (auto)"}, brightest numbered; zoom in for more`}
          />
        </Stack>
      </Panel>

      <Panel
        title="Basis"
        defaultOpen={false}
        colors={colors}
        blocked={hasPeaks ? undefined : "Run Detection before choosing basis vectors."}
      >
        <Stack direction="row" spacing={2} alignItems="flex-start" flexWrap="wrap">
          {controlsVisible && (
          <Stack spacing={1.5} sx={{ minWidth: 220 }}>
            <NumberField
              label="num_candidates"
              hint="0 picks the count from the busiest scan position"
              value={numCandidates}
              onChange={setNumCandidates}
              colors={colors}
            />
            <NumberField label="min_spacing" value={candidateMinSpacing} step={0.5} onChange={setCandidateMinSpacing} colors={colors} />
            <NumberField label="min_abs_intensity" value={candidateMinAbsIntensity} step={0.1} onChange={setCandidateMinAbsIntensity} colors={colors} />

            <Stack direction="row" spacing={0.5}>
              {BASIS_LABELS.map((role, i) => (
                <Button
                  key={role}
                  size="small"
                  sx={{
                    ...compactButton,
                    color: activeRole === role ? BASIS_COLORS[i] : colors.textMuted,
                    fontWeight: activeRole === role ? 700 : 400,
                  }}
                  onClick={() => setActiveRole(role)}
                >
                  {role}
                </Button>
              ))}
            </Stack>

            <Button
              size="small"
              sx={{ ...compactButton, color: colors.textMuted, alignSelf: "flex-start" }}
              onClick={() => {
                setOriginIndex(-1);
                setG1Index(-1);
                setG2Index(-1);
                setOriginRc([]);
                setG1Rc([]);
                setG2Rc([]);
              }}
            >
              Reset Basis
            </Button>

            <Box sx={{ fontFamily: "monospace", fontSize: 11 }}>
              <Typography sx={{ fontSize: 11, fontFamily: "monospace", color: BASIS_COLORS[0] }}>
                origin {vectorText(originRc)}{originIndex >= 0 ? ` · candidate ${originIndex}` : ""}
              </Typography>
              <Typography sx={{ fontSize: 11, fontFamily: "monospace", color: BASIS_COLORS[1] }}>
                g1 {vectorText(g1Rc)}{g1Index >= 0 ? ` · candidate ${g1Index}` : ""}
              </Typography>
              <Typography sx={{ fontSize: 11, fontFamily: "monospace", color: BASIS_COLORS[2] }}>
                g2 {vectorText(g2Rc)}{g2Index >= 0 ? ` · candidate ${g2Index}` : ""}
              </Typography>
            </Box>
          </Stack>
          )}

          <ImageCanvas
            bytes={bvmPng}
            shape={qShape}
            colors={colors}
            overlay={drawCandidates}
            markers={basisMarkers}
            onMoveMarker={(i, row, col) => setDragPreview([i, row, col])}
            onCommitMarker={setRoleVector}
            onPick={pickCandidate}
            caption={`click a numbered candidate to set ${activeRole}, or drag a marker`}
          />
        </Stack>
      </Panel>

      <Panel
        title="Lattice fit"
        defaultOpen={false}
        colors={colors}
        blocked={basisMarkers.length === 3 ? undefined : "Choose the basis vectors before fitting."}
      >
        {controlsVisible && (
        <Stack direction="row" spacing={1.5} alignItems="center" flexWrap="wrap" useFlexGap>
          <NumberField label="min_num_peaks" value={minNumPeaks} onChange={setMinNumPeaks} colors={colors} />
          <NumberField
            label="max_peak_shift"
            hint="inclusion radius in pixels; 0 uses half the shorter spacing"
            value={maxPeakShift}
            step={0.5}
            onChange={setMaxPeakShift}
            colors={colors}
          />
          <Button
            size="small"
            sx={{ ...compactButton, color: colors.accent }}
            disabled={busy}
            onClick={() => send({ type: "run_fit" })}
          >
            {fitting ? "Fitting..." : "Run Fit"}
          </Button>
        </Stack>
        )}

        {fitState === "done" && (
          <Stack direction="row" spacing={2} sx={{ mt: `${SPACING.MD}px` }} flexWrap="wrap">
            <ImageCanvas bytes={maskWeightPng} shape={scanShape} colors={colors} caption="mask weight" />
            <ImageCanvas bytes={fitErrorPng} shape={scanShape} colors={colors} caption="fit error (px)" />
          </Stack>
        )}
      </Panel>
      </div>
    </Box>
  );
}

export const render = createRender(ShowBragg);
