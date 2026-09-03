package dev.riszn.androidtrainer;

import android.app.Activity;
import android.app.ActivityManager;
import android.app.ApplicationExitInfo;
import android.content.ContentValues;
import android.content.Intent;
import android.content.res.ColorStateList;
import android.net.Uri;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.PowerManager;
import android.provider.MediaStore;
import android.provider.Settings;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONObject;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.security.MessageDigest;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public final class MainActivity extends Activity {
    private static final int REQ_BUNDLE = 1001;
    private static final int REQ_REPORT = 1002;
    private static final int REQ_CRASH_TRACE = 1003;
    private static final int REQ_PILOT = 1004;
    private static final int REQ_STAGE = 1005;
    private static final int REQ_CHECKPOINT_EXPORT = 1006;
    private static final int REQ_F2_PILOT = 1007;
    private static final int REQ_F2_STAGE = 1008;

    static {
        System.loadLibrary("android_trainer");
    }

    private TextView log;
    private ScrollView logScroller;
    private ScrollView pageScroller;
    private Button run;
    private Button pilotRun;
    private Button productionRun;
    private Button exportCheckpoint;
    private Button f2PilotRun;
    private Button f2ProductionRun;
    private File bundleDir;
    private File pilotDir;
    private File stageDir;
    private File f2PilotDir;
    private File f2StageDir;
    private String f2PilotSchema = "";
    private String f2StageSchema = "";
    private String bundleModelStateSha = "";
    private File lastTrainingCheckpoint;
    private volatile boolean trainingActive = false;
    private String lastReport = "";
    private File lastNativeTrace;
    private PowerManager powerManager;

    private static native String nativeProbe();
    private static native String nativeOpenClConformance();
    private static native String nativeValidateBundle(String bundleDir);
    private static native String nativeRunGate(String bundleDir, String workDir, float thermalHeadroom);
    private static native String nativeRunLrPilot(String bundleDir, String pilotDir, String workDir);
    private static native String nativeRunStage(String bundleDir, String stageDir, String workDir);
    private static native String nativeRunF2SftLrPilot(String bundleDir, String pilotDir, String workDir);
    private static native String nativeRunF2SftStage(String bundleDir, String stageDir, String workDir);

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        powerManager = (PowerManager) getSystemService(POWER_SERVICE);

        // The whole screen is scrollable. The previous layout placed the long
        // workflow above a weighted log panel, which could collapse the log to
        // almost zero height on phones. Keep the workflow compact and give the
        // output console a real, independently scrollable viewport.
        pageScroller = new ScrollView(this);
        pageScroller.setFillViewport(true);
        pageScroller.setVerticalScrollBarEnabled(true);
        pageScroller.setScrollbarFadingEnabled(false);
        pageScroller.setBackgroundColor(0xff0d0f12);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(18), dp(18), dp(18), dp(28));
        pageScroller.addView(
                root,
                new ScrollView.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT));

        LinearLayout header = card();
        header.setPadding(dp(18), dp(16), dp(18), dp(16));

        TextView title = new TextView(this);
        title.setText("MODEL #0001  •  TRAINER");
        title.setTextColor(0xfff5f7fa);
        title.setTextSize(24);
        title.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        title.setGravity(Gravity.CENTER_VERTICAL);
        header.addView(title, new LinearLayout.LayoutParams(-1, -2));

        TextView subtitle = new TextView(this);
        subtitle.setText("Native OpenCL FP32 • Foundation + F2 SFT workflow");
        subtitle.setTextColor(0xff9da4ad);
        subtitle.setTextSize(13);
        subtitle.setPadding(0, dp(6), 0, 0);
        header.addView(subtitle, new LinearLayout.LayoutParams(-1, -2));

        root.addView(header, sectionParams(dp(0), dp(12)));

        LinearLayout workflowHeader = new LinearLayout(this);
        workflowHeader.setOrientation(LinearLayout.HORIZONTAL);
        workflowHeader.setGravity(Gravity.CENTER_VERTICAL);
        workflowHeader.setPadding(dp(4), dp(4), dp(4), dp(8));

        TextView workflowTitle = new TextView(this);
        workflowTitle.setText("WORKFLOW STEPS");
        workflowTitle.setTextColor(0xffc7cdd4);
        workflowTitle.setTextSize(12);
        workflowTitle.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        workflowTitle.setLetterSpacing(0.08f);
        workflowHeader.addView(
                workflowTitle,
                new LinearLayout.LayoutParams(0, -2, 1f));

        TextView scrollHint = pill("↕  Scroll");
        workflowHeader.addView(scrollHint);

        root.addView(workflowHeader, new LinearLayout.LayoutParams(-1, -2));

        LinearLayout buttons = new LinearLayout(this);
        buttons.setOrientation(LinearLayout.VERTICAL);

        Button probe = button("1. Probe native OpenCL device");
        probe.setOnClickListener(v -> background("PROBE", () -> nativeProbe()));
        buttons.addView(probe);

        Button nativeGate = button("2. Native OpenCL primitive conformance");
        nativeGate.setOnClickListener(v -> background(
                "NATIVE OPENCL CONFORMANCE", () -> nativeOpenClConformance()));
        buttons.addView(nativeGate);

        Button select = button("3. Import local .atb bundle");
        select.setOnClickListener(v -> selectBundle());
        buttons.addView(select);

        run = button("4. Run full native OpenCL training gate");
        run.setEnabled(false);
        run.setOnClickListener(v -> runGate());
        buttons.addView(run);

        Button selectPilot = button("5. Import Foundation-v3 LR pilot package");
        selectPilot.setOnClickListener(v -> selectPilot());
        buttons.addView(selectPilot);

        pilotRun = button("6. Run Foundation-v3 LR pilot");
        pilotRun.setEnabled(false);
        pilotRun.setOnClickListener(v -> runLrPilot());
        buttons.addView(pilotRun);

        Button selectStage = button("7. Import locked Foundation-v3 .atstage");
        selectStage.setOnClickListener(v -> selectStage());
        buttons.addView(selectStage);

        productionRun = button("8. Run / resume Foundation-v3 training");
        productionRun.setEnabled(false);
        productionRun.setOnClickListener(v -> runProductionStage());
        buttons.addView(productionRun);

        exportCheckpoint = button("9. Export final native checkpoint");
        exportCheckpoint.setEnabled(false);
        exportCheckpoint.setOnClickListener(v -> exportTrainingCheckpoint());
        buttons.addView(exportCheckpoint);

        Button selectF2Pilot = button("10. Import masked SFT LR pilot package");
        selectF2Pilot.setOnClickListener(v -> selectF2Pilot());
        buttons.addView(selectF2Pilot);

        f2PilotRun = button("11. Run assistant-only LR pilot");
        f2PilotRun.setEnabled(false);
        f2PilotRun.setOnClickListener(v -> runF2SftLrPilot());
        buttons.addView(f2PilotRun);

        Button selectF2Stage = button("12. Import locked masked SFT stage");
        selectF2Stage.setOnClickListener(v -> selectF2Stage());
        buttons.addView(selectF2Stage);

        f2ProductionRun = button("13. Run / resume masked SFT training");
        f2ProductionRun.setEnabled(false);
        f2ProductionRun.setOnClickListener(v -> runF2SftStage());
        buttons.addView(f2ProductionRun);

        root.addView(buttons, new LinearLayout.LayoutParams(-1, -2));

        LinearLayout outputCard = card();
        LinearLayout.LayoutParams outputCardLp = sectionParams(dp(14), dp(12));

        LinearLayout outputHeader = new LinearLayout(this);
        outputHeader.setOrientation(LinearLayout.HORIZONTAL);
        outputHeader.setGravity(Gravity.CENTER_VERTICAL);
        outputHeader.setPadding(dp(14), dp(12), dp(14), dp(8));

        TextView outputTitle = new TextView(this);
        outputTitle.setText(">_  LIVE OUTPUT");
        outputTitle.setTextColor(0xffe6e9ed);
        outputTitle.setTextSize(13);
        outputTitle.setTypeface(Typeface.MONOSPACE, Typeface.BOLD);
        outputHeader.addView(
                outputTitle,
                new LinearLayout.LayoutParams(0, -2, 1f));

        TextView live = pill("●  Live");
        live.setTextColor(0xff73e59b);
        outputHeader.addView(live);

        outputCard.addView(outputHeader, new LinearLayout.LayoutParams(-1, -2));

        log = new TextView(this);
        log.setTextColor(0xffd9dde3);
        log.setTextSize(11.5f);
        log.setTextIsSelectable(true);
        log.setTypeface(Typeface.MONOSPACE);
        log.setLineSpacing(0f, 1.12f);
        log.setPadding(dp(14), dp(8), dp(14), dp(14));
        log.setText("Waiting for output…\n");

        logScroller = new ScrollView(this);
        logScroller.setFillViewport(true);
        logScroller.setVerticalScrollBarEnabled(true);
        logScroller.setScrollbarFadingEnabled(false);
        logScroller.setNestedScrollingEnabled(true);
        logScroller.setBackgroundColor(0xff0a0c0f);
        logScroller.addView(
                log,
                new ScrollView.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT));

        LinearLayout.LayoutParams logLp =
                new LinearLayout.LayoutParams(-1, dp(260));
        logLp.setMargins(dp(10), 0, dp(10), dp(12));
        outputCard.addView(logScroller, logLp);
        root.addView(outputCard, outputCardLp);

        TextView outputHint = new TextView(this);
        outputHint.setText(
                "Output stays here while the workflow runs. Swipe inside the console "
                + "for older lines; swipe outside it to move through the page.");
        outputHint.setTextColor(0xff7f8791);
        outputHint.setTextSize(11);
        outputHint.setPadding(dp(4), 0, dp(4), dp(8));
        root.addView(outputHint, new LinearLayout.LayoutParams(-1, -2));

        Button export = actionButton("⇩  Export last JSON report");
        export.setOnClickListener(v -> exportReport());
        root.addView(export);

        Button exportCrash = actionButton("⇩  Export native crash trace");
        exportCrash.setOnClickListener(v -> exportNativeTrace());
        root.addView(exportCrash);

        setContentView(pageScroller);
        showPreviousExitDiagnostics();
        restoreExistingPackages();
        handleLaunchIntent(getIntent());
    }

    @Override protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        handleLaunchIntent(intent);
    }

    private void handleLaunchIntent(Intent intent) {
        if (intent == null || !Intent.ACTION_VIEW.equals(intent.getAction())) return;
        Uri data = intent.getData();
        if (data == null) return;
        if ("androidtrainer".equals(data.getScheme()) &&
                "gate".equals(data.getHost()) && "/run".equals(data.getPath())) {
            File existing = new File(getFilesDir(), "gate_bundle");
            new Thread(() -> {
                try {
                    verifyBundleFiles(existing);
                    String check = nativeValidateBundle(existing.getAbsolutePath());
                    if (!"PASS".equals(new JSONObject(check).optString("status"))) {
                        throw new IllegalStateException(check);
                    }
                    bundleDir = existing;
                    runOnUiThread(this::runGate);
                } catch (Throwable t) {
                    append("DEEP-LINK GATE FAIL: " + t);
                }
            }, "android-trainer-deep-link").start();
            return;
        }
        importBundle(data, true);
    }

    private void showPreviousExitDiagnostics() {
        if (Build.VERSION.SDK_INT < 30) return;
        try {
            ActivityManager am = (ActivityManager) getSystemService(ACTIVITY_SERVICE);
            if (am == null) return;
            List<ApplicationExitInfo> infos = am.getHistoricalProcessExitReasons(null, 0, 8);
            ApplicationExitInfo chosen = null;
            for (ApplicationExitInfo info : infos) {
                int r = info.getReason();
                if (r == ApplicationExitInfo.REASON_CRASH_NATIVE ||
                    r == ApplicationExitInfo.REASON_CRASH ||
                    r == ApplicationExitInfo.REASON_SIGNALED ||
                    r == ApplicationExitInfo.REASON_LOW_MEMORY ||
                    r == ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE ||
                    r == ApplicationExitInfo.REASON_ANR) {
                    chosen = info;
                    break;
                }
            }
            File stageFile = new File(getFilesDir(), "last_native_stage.txt");
            String stage = stageFile.isFile()
                    ? new String(java.nio.file.Files.readAllBytes(stageFile.toPath()),
                                 java.nio.charset.StandardCharsets.UTF_8)
                    : "(none)";
            append("\n=== PREVIOUS PROCESS EXIT ===");
            append("last_native_stage: " + stage);
            if (chosen == null) {
                append("no crash/kill record found in ApplicationExitInfo");
                return;
            }
            append("reason: " + exitReasonName(chosen.getReason()) + " (" + chosen.getReason() + ")");
            append("status/signal: " + chosen.getStatus());
            append("timestamp_ms: " + chosen.getTimestamp());
            append("pss_kb: " + chosen.getPss());
            append("rss_kb: " + chosen.getRss());
            String d = chosen.getDescription();
            if (d != null && !d.isEmpty()) append("description: " + d);

            // Android 12+ exposes this app's native crash tombstone as the
            // platform Tombstone protobuf. Capture it verbatim; do not parse
            // or transform it in-process, so the forensic artifact stays exact.
            if (Build.VERSION.SDK_INT >= 31 &&
                chosen.getReason() == ApplicationExitInfo.REASON_CRASH_NATIVE) {
                try (InputStream trace = chosen.getTraceInputStream()) {
                    if (trace == null) {
                        append("native_tombstone: unavailable_or_overwritten");
                    } else {
                        File tomb = new File(
                                getFilesDir(),
                                "native_tombstone_" + chosen.getTimestamp() + ".pb");
                        try (OutputStream out = new BufferedOutputStream(new FileOutputStream(tomb))) {
                            copy(trace, out);
                        }
                        lastNativeTrace = tomb;
                        append("native_tombstone_bytes: " + tomb.length());
                        append("native_tombstone_sha256: " + sha256(tomb));
                        append("native_tombstone_ready: YES");
                    }
                } catch (Throwable t) {
                    append("native_tombstone_capture_error: " + t);
                }
            }
        } catch (Throwable t) {
            append("exit diagnostic unavailable: " + t);
        }
    }

    private static String exitReasonName(int r) {
        if (Build.VERSION.SDK_INT >= 30) {
            switch (r) {
                case ApplicationExitInfo.REASON_CRASH_NATIVE: return "CRASH_NATIVE";
                case ApplicationExitInfo.REASON_CRASH: return "CRASH_JAVA";
                case ApplicationExitInfo.REASON_SIGNALED: return "SIGNALED";
                case ApplicationExitInfo.REASON_LOW_MEMORY: return "LOW_MEMORY";
                case ApplicationExitInfo.REASON_ANR: return "ANR";
                case ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE: return "EXCESSIVE_RESOURCE_USAGE";
                case ApplicationExitInfo.REASON_EXIT_SELF: return "EXIT_SELF";
                case ApplicationExitInfo.REASON_USER_REQUESTED: return "USER_REQUESTED";
                case ApplicationExitInfo.REASON_PACKAGE_UPDATED: return "PACKAGE_UPDATED";
                default: return "REASON_" + r;
            }
        }
        return "UNKNOWN";
    }

    private int dp(int value) {
        return Math.round(
                value * getResources().getDisplayMetrics().density);
    }

    private GradientDrawable roundedBackground(
            int fillColor, int strokeColor, int radiusDp) {
        GradientDrawable bg = new GradientDrawable();
        bg.setColor(fillColor);
        bg.setCornerRadius(dp(radiusDp));
        if (strokeColor != 0) bg.setStroke(dp(1), strokeColor);
        return bg;
    }

    private LinearLayout card() {
        LinearLayout card = new LinearLayout(this);
        card.setOrientation(LinearLayout.VERTICAL);
        card.setBackground(
                roundedBackground(0xff15181d, 0xff292e36, 16));
        if (Build.VERSION.SDK_INT >= 21) card.setElevation(dp(1));
        return card;
    }

    private LinearLayout.LayoutParams sectionParams(
            int topMargin, int bottomMargin) {
        LinearLayout.LayoutParams lp =
                new LinearLayout.LayoutParams(-1, -2);
        lp.setMargins(0, topMargin, 0, bottomMargin);
        return lp;
    }

    private TextView pill(String text) {
        TextView v = new TextView(this);
        v.setText(text);
        v.setTextColor(0xffaab1ba);
        v.setTextSize(11);
        v.setGravity(Gravity.CENTER);
        v.setPadding(dp(10), dp(6), dp(10), dp(6));
        v.setBackground(
                roundedBackground(0xff1c2026, 0xff2b3139, 99));
        return v;
    }

    private Button button(String text) {
        Button b = new Button(this);
        b.setText(text);
        b.setAllCaps(false);
        b.setTextSize(13.5f);
        b.setGravity(Gravity.CENTER_VERTICAL | Gravity.START);
        b.setPadding(dp(16), 0, dp(14), 0);
        b.setMinHeight(0);
        b.setMinimumHeight(0);
        b.setBackgroundTintList(new ColorStateList(
                new int[][]{
                    new int[]{-android.R.attr.state_enabled},
                    new int[]{android.R.attr.state_pressed},
                    new int[]{}
                },
                new int[]{
                    0xff1d2025,
                    0xff343a43,
                    0xff2a2f36
                }));
        b.setTextColor(new ColorStateList(
                new int[][]{
                    new int[]{-android.R.attr.state_enabled},
                    new int[]{}
                },
                new int[]{
                    0xff666d76,
                    0xfff0f2f5
                }));
        LinearLayout.LayoutParams lp =
                new LinearLayout.LayoutParams(-1, dp(54));
        lp.setMargins(0, dp(4), 0, dp(4));
        b.setLayoutParams(lp);
        return b;
    }

    private Button actionButton(String text) {
        Button b = button(text);
        b.setGravity(Gravity.CENTER);
        LinearLayout.LayoutParams lp =
                new LinearLayout.LayoutParams(-1, dp(56));
        lp.setMargins(0, dp(4), 0, dp(4));
        b.setLayoutParams(lp);
        return b;
    }

    private void append(String s) {
        runOnUiThread(() -> {
            if (log == null) return;
            log.append(s + "\n");
            if (logScroller != null) {
                logScroller.post(() ->
                        logScroller.fullScroll(View.FOCUS_DOWN));
            }
        });
    }

    private void background(String label, Job job) {
        append("\n=== " + label + " ===");
        new Thread(() -> {
            try {
                String out = job.run();
                append(pretty(out));
            } catch (Throwable t) {
                append("FAIL: " + t);
            }
        }, "android-trainer-" + label.toLowerCase(Locale.ROOT)).start();
    }

    private interface Job { String run() throws Exception; }

    private String pretty(String raw) {
        try { return new JSONObject(raw).toString(2); }
        catch (Exception ignored) { return raw; }
    }

    private void selectBundle() {
        Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        i.addCategory(Intent.CATEGORY_OPENABLE);
        i.setType("*/*");
        i.putExtra(Intent.EXTRA_MIME_TYPES, new String[]{"application/zip", "application/octet-stream"});
        startActivityForResult(i, REQ_BUNDLE);
    }

    @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK || data == null || data.getData() == null) return;
        if (requestCode == REQ_BUNDLE) importBundle(data.getData());
        if (requestCode == REQ_PILOT) importPilot(data.getData());
        if (requestCode == REQ_STAGE) importStage(data.getData());
        if (requestCode == REQ_REPORT) writeReport(data.getData());
        if (requestCode == REQ_CRASH_TRACE) writeNativeTrace(data.getData());
        if (requestCode == REQ_CHECKPOINT_EXPORT)
            writeTrainingCheckpoint(data.getData());
        if (requestCode == REQ_F2_PILOT)
            importF2Pilot(data.getData());
        if (requestCode == REQ_F2_STAGE)
            importF2Stage(data.getData());
    }

    private void importBundle(Uri uri) {
        importBundle(uri, false);
    }

    private void importBundle(Uri uri, boolean runAfterImport) {
        append("\n=== IMPORT BUNDLE ===");
        run.setEnabled(false);
        new Thread(() -> {
            try {
                File zip = new File(getFilesDir(), "model0001-gate.atb");
                try (InputStream in = getContentResolver().openInputStream(uri);
                     OutputStream out = new BufferedOutputStream(new FileOutputStream(zip))) {
                    if (in == null) throw new IllegalStateException("content resolver returned null stream");
                    copy(in, out);
                }
                String bundleSha = sha256(zip);
                File dest = new File(getFilesDir(), "gate_bundle");
                deleteTree(dest);
                if (!dest.mkdirs() && !dest.isDirectory()) throw new IllegalStateException("cannot create bundle dir");
                unzipSafely(zip, dest);
                verifyBundleFiles(dest);
                String nativeCheck = nativeValidateBundle(dest.getAbsolutePath());
                append("bundle_sha256: " + bundleSha);
                append(pretty(nativeCheck));
                JSONObject nativeResult = new JSONObject(nativeCheck);
                if (!"PASS".equals(nativeResult.optString("status"))) {
                    throw new IllegalStateException(
                            "native bundle validation failed: " + nativeCheck);
                }
                bundleDir = dest;
                bundleModelStateSha = readBundleModelStateSha(dest);
                append("bundle_model_state_sha256: " + bundleModelStateSha);
                runOnUiThread(() -> {
                    updateAllModeButtons();
                    if (runAfterImport && isCptV2Source()) runGate();
                });
            } catch (Throwable t) {
                append("IMPORT FAIL: " + t);
            }
        }, "android-trainer-import").start();
    }


    private void selectPilot() {
        Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        i.addCategory(Intent.CATEGORY_OPENABLE);
        i.setType("*/*");
        i.putExtra(Intent.EXTRA_MIME_TYPES,
                new String[]{"application/zip", "application/octet-stream"});
        startActivityForResult(i, REQ_PILOT);
    }

    private void importPilot(Uri uri) {
        append("\n=== IMPORT FOUNDATION-v3 LR PILOT ===");
        pilotRun.setEnabled(false);
        new Thread(() -> {
            try {
                File zip = new File(getFilesDir(), "model0001-v3-lr-pilot.atpilot");
                try (InputStream in = getContentResolver().openInputStream(uri);
                     OutputStream out = new BufferedOutputStream(new FileOutputStream(zip))) {
                    if (in == null)
                        throw new IllegalStateException("content resolver returned null pilot stream");
                    copy(in, out);
                }
                String pilotSha = sha256(zip);
                File dest = new File(getFilesDir(), "pilot_bundle");
                deleteTree(dest);
                if (!dest.mkdirs() && !dest.isDirectory())
                    throw new IllegalStateException("cannot create pilot dir");
                unzipSafely(zip, dest);
                verifyPilotFiles(dest);
                pilotDir = dest;
                append("pilot_package_sha256: " + pilotSha);
                append("pilot package verification: PASS");
                runOnUiThread(this::updatePilotEnabled);
            } catch (Throwable t) {
                append("PILOT IMPORT FAIL: " + t);
            }
        }, "android-trainer-pilot-import").start();
    }

    private void verifyPilotFiles(File root) throws Exception {
        File manifestFile = new File(root, "manifest.json");
        if (!manifestFile.isFile())
            throw new IllegalStateException("pilot manifest.json missing");
        String text = new String(
                java.nio.file.Files.readAllBytes(manifestFile.toPath()),
                java.nio.charset.StandardCharsets.UTF_8);
        JSONObject manifest = new JSONObject(text);
        if (!"model0001_v3_lr_pilot_v1".equals(manifest.getString("schema")))
            throw new IllegalStateException("wrong pilot schema");
        if (!"047b0f6ec18046c7a5ae7da707e91a03e26a6819cfec254f8ad541c8ddbf696d"
                .equals(manifest.getString("source_model_state_sha256")))
            throw new IllegalStateException("pilot source-model SHA mismatch");
        JSONObject guards = manifest.getJSONObject("hard_guards");
        if (guards.getBoolean("test_split_packaged"))
            throw new IllegalStateException("pilot package contains test split");
        if (guards.getBoolean("dataset_v2_train_bin_packaged"))
            throw new IllegalStateException("pilot package contains Dataset-v2 train");

        JSONObject data = manifest.getJSONObject("data");
        String[] keys = new String[]{"v3_train", "v3_validation", "v1_validation"};
        for (String key : keys) {
            JSONObject spec = data.getJSONObject(key);
            File file = canonicalChild(root, spec.getString("path"));
            if (!file.isFile())
                throw new IllegalStateException("pilot data missing: " + key);
            long expectedBytes = spec.getLong("uint16_tokens") * 2L;
            if (file.length() != expectedBytes)
                throw new IllegalStateException("pilot data size mismatch: " + key);
            String got = sha256(file);
            if (!got.equalsIgnoreCase(spec.getString("sha256")))
                throw new IllegalStateException("pilot data SHA mismatch: " + key);
        }
    }

    private boolean isCptV2Source() {
        return "047b0f6ec18046c7a5ae7da707e91a03e26a6819cfec254f8ad541c8ddbf696d"
                .equals(bundleModelStateSha);
    }

    private boolean isFoundationV3Source() {
        return "10836dbde12e6c1eb732c1b6695ed248af5754d038011058250e81593287d00b"
                .equals(bundleModelStateSha);
    }

    private void updatePilotEnabled() {
        if (pilotRun != null)
            pilotRun.setEnabled(
                    bundleDir != null && pilotDir != null && isCptV2Source());
    }



    private void selectF2Pilot() {
        Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        i.addCategory(Intent.CATEGORY_OPENABLE);
        i.setType("*/*");
        i.putExtra(Intent.EXTRA_MIME_TYPES,
                new String[]{"application/zip", "application/octet-stream"});
        startActivityForResult(i, REQ_F2_PILOT);
    }

    private void importF2Pilot(Uri uri) {
        append("\n=== IMPORT F2 SFT LR PILOT ===");
        if (f2PilotRun != null) f2PilotRun.setEnabled(false);
        new Thread(() -> {
            try {
                File zip = new File(
                        getFilesDir(), "model0001-f2-sft-lr-pilot.atsftpilot");
                try (InputStream in = getContentResolver().openInputStream(uri);
                     OutputStream out =
                             new BufferedOutputStream(new FileOutputStream(zip))) {
                    if (in == null)
                        throw new IllegalStateException(
                                "content resolver returned null F2 pilot stream");
                    copy(in, out);
                }
                String packageSha = sha256(zip);
                File dest = new File(getFilesDir(), "f2_sft_pilot_bundle");
                deleteTree(dest);
                if (!dest.mkdirs() && !dest.isDirectory())
                    throw new IllegalStateException("cannot create F2 pilot dir");
                unzipSafely(zip, dest);
                f2PilotSchema = verifyF2PilotFiles(dest);
                f2PilotDir = dest;
                append("f2_sft_pilot_sha256: " + packageSha);
                append("F2 SFT pilot package verification: PASS");
                runOnUiThread(this::updateF2PilotEnabled);
            } catch (Throwable t) {
                append("F2 PILOT IMPORT FAIL: " + t);
            }
        }, "android-trainer-f2-pilot-import").start();
    }

    private String verifyF2PilotFiles(File root) throws Exception {
        File manifestFile = new File(root, "manifest.json");
        if (!manifestFile.isFile())
            throw new IllegalStateException("masked SFT pilot manifest.json missing");
        String text = new String(
                java.nio.file.Files.readAllBytes(manifestFile.toPath()),
                java.nio.charset.StandardCharsets.UTF_8);
        JSONObject manifest = new JSONObject(text);
        String schema = manifest.getString("schema");
        boolean repair = "model0001_f2r_lr_pilot_v1".equals(schema);
        if (!"model0001_f2_sft_lr_pilot_v1".equals(schema) && !repair)
            throw new IllegalStateException("wrong masked SFT pilot schema");
        if (!"10836dbde12e6c1eb732c1b6695ed248af5754d038011058250e81593287d00b"
                .equals(manifest.getString("source_model_state_sha256")))
            throw new IllegalStateException("masked SFT pilot source-model SHA mismatch");
        if (!"assistant_content_only_cross_entropy".equals(
                manifest.getString("objective")))
            throw new IllegalStateException("masked SFT pilot objective drift");

        JSONObject guards = manifest.getJSONObject("hard_guards");
        if (!guards.getBoolean("assistant_only_loss"))
            throw new IllegalStateException("masked SFT pilot mask guard missing");
        if (guards.getBoolean("test_split_packaged"))
            throw new IllegalStateException("masked SFT pilot contains test split");
        if (guards.getBoolean("dataset_v2_train_bin_packaged"))
            throw new IllegalStateException("masked SFT pilot contains Dataset-v2 train");
        if (guards.getBoolean("foundation_v3_train_bin_packaged"))
            throw new IllegalStateException("masked SFT pilot contains Foundation-v3 train");
        if (repair) {
            if (!"friend_f2r_repair_sft".equals(
                    manifest.getString("stage_objective")))
                throw new IllegalStateException("F2R pilot stage objective drift");
            if (!guards.getBoolean("record_isolated_packing"))
                throw new IllegalStateException("F2R record-isolated guard missing");
            if (guards.getInt("cross_record_windows") != 0)
                throw new IllegalStateException("F2R cross-record windows detected");
            if (guards.getBoolean("behavior_eval_prompts_used_for_train"))
                throw new IllegalStateException("F2R behavior-eval prompt leak");
        }

        JSONObject data = manifest.getJSONObject("data");
        String[] masked = new String[]{"sft_train", "sft_validation"};
        for (String key : masked) {
            JSONObject spec = data.getJSONObject(key);
            int windows = spec.getInt("windows");
            File tokens = canonicalChild(root, spec.getString("tokens_path"));
            File mask = canonicalChild(root, spec.getString("mask_path"));
            if (!tokens.isFile() || !mask.isFile())
                throw new IllegalStateException("masked SFT pilot data missing: " + key);
            if (tokens.length() != (long) windows * 257L * 2L)
                throw new IllegalStateException("masked SFT token-window size mismatch: " + key);
            if (mask.length() != (long) windows * 256L)
                throw new IllegalStateException("masked SFT mask-window size mismatch: " + key);
            if (!sha256(tokens).equalsIgnoreCase(spec.getString("tokens_sha256")))
                throw new IllegalStateException("masked SFT tokens SHA mismatch: " + key);
            if (!sha256(mask).equalsIgnoreCase(spec.getString("mask_sha256")))
                throw new IllegalStateException("masked SFT mask SHA mismatch: " + key);
        }
        for (String key : new String[]{"v3_validation", "v1_validation"}) {
            JSONObject spec = data.getJSONObject(key);
            File file = canonicalChild(root, spec.getString("path"));
            if (!file.isFile())
                throw new IllegalStateException("retention validation missing: " + key);
            if (file.length() != spec.getLong("uint16_tokens") * 2L)
                throw new IllegalStateException("retention data size mismatch: " + key);
            if (!sha256(file).equalsIgnoreCase(spec.getString("sha256")))
                throw new IllegalStateException("retention SHA mismatch: " + key);
        }
        return schema;
    }


    private void selectF2Stage() {
        Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        i.addCategory(Intent.CATEGORY_OPENABLE);
        i.setType("*/*");
        i.putExtra(Intent.EXTRA_MIME_TYPES,
                new String[]{"application/zip", "application/octet-stream"});
        startActivityForResult(i, REQ_F2_STAGE);
    }

    private void importF2Stage(Uri uri) {
        append("\n=== IMPORT MASKED SFT PRODUCTION STAGE ===");
        if (f2ProductionRun != null) f2ProductionRun.setEnabled(false);
        new Thread(() -> {
            try {
                File zip = new File(
                        getFilesDir(), "model0001-f2-sft.atsftstage");
                try (InputStream in = getContentResolver().openInputStream(uri);
                     OutputStream out =
                             new BufferedOutputStream(new FileOutputStream(zip))) {
                    if (in == null)
                        throw new IllegalStateException(
                                "content resolver returned null F2 stage stream");
                    copy(in, out);
                }
                String packageSha = sha256(zip);
                File dest = new File(getFilesDir(), "f2_sft_stage_bundle");
                deleteTree(dest);
                if (!dest.mkdirs() && !dest.isDirectory())
                    throw new IllegalStateException("cannot create F2 stage dir");
                unzipSafely(zip, dest);
                f2StageSchema = verifyF2StageFiles(dest);
                f2StageDir = dest;
                append("f2_sft_stage_sha256: " + packageSha);
                append("masked SFT production stage verification: PASS");
                runOnUiThread(this::updateF2ProductionEnabled);
            } catch (Throwable t) {
                append("F2 STAGE IMPORT FAIL: " + t);
            }
        }, "android-trainer-f2-stage-import").start();
    }

    private String verifyF2StageFiles(File root) throws Exception {
        File manifestFile = new File(root, "manifest.json");
        if (!manifestFile.isFile())
            throw new IllegalStateException("masked SFT stage manifest.json missing");
        String text = new String(
                java.nio.file.Files.readAllBytes(manifestFile.toPath()),
                java.nio.charset.StandardCharsets.UTF_8);
        JSONObject manifest = new JSONObject(text);
        String schema = manifest.getString("schema");
        boolean repair = "model0001_f2r_stage_package_v1".equals(schema);
        if (!"model0001_f2_sft_stage_package_v1".equals(schema) && !repair)
            throw new IllegalStateException("wrong masked SFT stage schema");
        if (!"10836dbde12e6c1eb732c1b6695ed248af5754d038011058250e81593287d00b"
                .equals(manifest.getString("source_model_state_sha256")))
            throw new IllegalStateException("masked SFT stage source-model SHA mismatch");
        if (!"assistant_content_only_cross_entropy".equals(
                manifest.getString("objective")))
            throw new IllegalStateException("masked SFT stage objective drift");

        JSONObject guards = manifest.getJSONObject("hard_guards");
        if (!guards.getBoolean("assistant_only_loss"))
            throw new IllegalStateException("masked SFT assistant-only guard missing");
        if (guards.getBoolean("test_split_packaged"))
            throw new IllegalStateException("masked SFT stage contains test split");
        if (guards.getBoolean("foundation_v3_train_bin_packaged"))
            throw new IllegalStateException("masked SFT stage contains Foundation-v3 train");
        if (repair) {
            if (!guards.getBoolean("record_isolated_packing"))
                throw new IllegalStateException("F2R stage record-isolation guard missing");
            if (guards.getInt("cross_record_windows") != 0)
                throw new IllegalStateException("F2R stage cross-record windows detected");
            if (guards.getBoolean("behavior_eval_prompts_used_for_train"))
                throw new IllegalStateException("F2R stage behavior-eval prompt leak");
        }

        JSONObject recipe = manifest.getJSONObject("recipe");
        String expectedStage = repair ? "friend_f2r_repair_sft" : "friend_f2_sft";
        String expectedRecipeSchema = repair
                ? "model0001_f2r_stage_recipe_v1"
                : "model0001_f2_sft_stage_recipe_v1";
        if (!expectedStage.equals(recipe.getString("stage_name")))
            throw new IllegalStateException("masked SFT stage-name drift");
        if (!expectedRecipeSchema.equals(recipe.getString("schema")))
            throw new IllegalStateException("masked SFT recipe schema drift");
        if (!"fresh_zero_moments".equals(recipe.getString("optimizer_init")))
            throw new IllegalStateException("masked SFT optimizer-init drift");
        if (recipe.getBoolean("test_split_used"))
            throw new IllegalStateException("masked SFT recipe touches test split");

        JSONObject data = manifest.getJSONObject("data");
        for (String key : new String[]{"sft_train", "sft_validation"}) {
            JSONObject spec = data.getJSONObject(key);
            int windows = spec.getInt("windows");
            File tokens = canonicalChild(root, spec.getString("tokens_path"));
            File mask = canonicalChild(root, spec.getString("mask_path"));
            if (!tokens.isFile() || !mask.isFile())
                throw new IllegalStateException("masked SFT stage data missing: " + key);
            if (tokens.length() != (long) windows * 257L * 2L)
                throw new IllegalStateException("masked SFT stage tokens size mismatch: " + key);
            if (mask.length() != (long) windows * 256L)
                throw new IllegalStateException("masked SFT stage mask size mismatch: " + key);
            if (!sha256(tokens).equalsIgnoreCase(spec.getString("tokens_sha256")))
                throw new IllegalStateException("masked SFT stage tokens SHA mismatch: " + key);
            if (!sha256(mask).equalsIgnoreCase(spec.getString("mask_sha256")))
                throw new IllegalStateException("masked SFT stage mask SHA mismatch: " + key);
        }
        for (String key : new String[]{"v3_validation", "v1_validation"}) {
            JSONObject spec = data.getJSONObject(key);
            File file = canonicalChild(root, spec.getString("path"));
            if (!file.isFile() ||
                    file.length() != spec.getLong("uint16_tokens") * 2L)
                throw new IllegalStateException("masked SFT retention data mismatch: " + key);
            if (!sha256(file).equalsIgnoreCase(spec.getString("sha256")))
                throw new IllegalStateException("masked SFT retention SHA mismatch: " + key);
        }
        return schema;
    }

    private void selectStage() {
        Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        i.addCategory(Intent.CATEGORY_OPENABLE);
        i.setType("*/*");
        i.putExtra(Intent.EXTRA_MIME_TYPES,
                new String[]{"application/zip", "application/octet-stream"});
        startActivityForResult(i, REQ_STAGE);
    }

    private void importStage(Uri uri) {
        append("\n=== IMPORT FOUNDATION-v3 PRODUCTION STAGE ===");
        if (productionRun != null) productionRun.setEnabled(false);
        new Thread(() -> {
            try {
                File zip =
                        new File(getFilesDir(), "model0001-foundation-v3.atstage");
                try (InputStream in = getContentResolver().openInputStream(uri);
                     OutputStream out =
                             new BufferedOutputStream(new FileOutputStream(zip))) {
                    if (in == null)
                        throw new IllegalStateException(
                                "content resolver returned null stage stream");
                    copy(in, out);
                }
                String stageSha = sha256(zip);
                File dest = new File(getFilesDir(), "production_stage");
                deleteTree(dest);
                if (!dest.mkdirs() && !dest.isDirectory())
                    throw new IllegalStateException("cannot create stage dir");
                unzipSafely(zip, dest);
                verifyStageFiles(dest);
                stageDir = dest;
                append("stage_package_sha256: " + stageSha);
                append("production stage verification: PASS");
                runOnUiThread(this::updateProductionEnabled);
            } catch (Throwable t) {
                append("STAGE IMPORT FAIL: " + t);
            }
        }, "android-trainer-stage-import").start();
    }

    private void verifyStageFiles(File root) throws Exception {
        File manifestFile = new File(root, "manifest.json");
        if (!manifestFile.isFile())
            throw new IllegalStateException("stage manifest.json missing");
        String text = new String(
                java.nio.file.Files.readAllBytes(manifestFile.toPath()),
                java.nio.charset.StandardCharsets.UTF_8);
        JSONObject manifest = new JSONObject(text);
        if (!"model0001_native_stage_package_v2".equals(
                manifest.getString("schema")))
            throw new IllegalStateException("wrong stage schema");
        if (!"047b0f6ec18046c7a5ae7da707e91a03e26a6819cfec254f8ad541c8ddbf696d"
                .equals(manifest.getString("source_model_state_sha256")))
            throw new IllegalStateException("stage source-model SHA mismatch");
        JSONObject guards = manifest.getJSONObject("hard_guards");
        if (guards.getBoolean("test_split_packaged"))
            throw new IllegalStateException("stage contains test split");
        if (guards.getBoolean("dataset_v2_train_bin_packaged"))
            throw new IllegalStateException("stage contains Dataset-v2 train");

        JSONObject recipe = manifest.getJSONObject("recipe");
        if (!"friend_foundation_v3_cpt".equals(recipe.getString("stage_name")))
            throw new IllegalStateException("stage-name drift");
        if (recipe.getBoolean("test_split_used"))
            throw new IllegalStateException("recipe touches test split");
        if (!"fresh_zero_moments".equals(recipe.getString("optimizer_init")))
            throw new IllegalStateException("stage optimizer-init drift");

        JSONObject data = manifest.getJSONObject("data");
        String[] keys =
                new String[]{"train", "v3_validation", "v1_validation"};
        for (String key : keys) {
            JSONObject spec = data.getJSONObject(key);
            File file = canonicalChild(root, spec.getString("path"));
            if (!file.isFile())
                throw new IllegalStateException("stage data missing: " + key);
            long expectedBytes = spec.getLong("uint16_tokens") * 2L;
            if (file.length() != expectedBytes)
                throw new IllegalStateException(
                        "stage data size mismatch: " + key);
            if (!sha256(file).equalsIgnoreCase(spec.getString("sha256")))
                throw new IllegalStateException(
                        "stage data SHA mismatch: " + key);
        }
    }

    private void updateProductionEnabled() {
        if (productionRun != null)
            productionRun.setEnabled(
                    bundleDir != null && stageDir != null &&
                    !trainingActive && isCptV2Source());
    }

    private void updateF2PilotEnabled() {
        if (f2PilotRun != null)
            f2PilotRun.setEnabled(
                    bundleDir != null && f2PilotDir != null &&
                    isFoundationV3Source() && !trainingActive);
    }

    private void updateF2ProductionEnabled() {
        if (f2ProductionRun != null)
            f2ProductionRun.setEnabled(
                    bundleDir != null && f2StageDir != null &&
                    isFoundationV3Source() && !trainingActive);
    }

    private void updateAllModeButtons() {
        if (run != null)
            run.setEnabled(bundleDir != null && isCptV2Source());
        updatePilotEnabled();
        updateProductionEnabled();
        updateF2PilotEnabled();
        updateF2ProductionEnabled();
    }

    private void restoreExistingPackages() {
        new Thread(() -> {
            try {
                File existingBundle = new File(getFilesDir(), "gate_bundle");
                if (existingBundle.isDirectory()) {
                    verifyBundleFiles(existingBundle);
                    String check =
                            nativeValidateBundle(existingBundle.getAbsolutePath());
                    if ("PASS".equals(
                            new JSONObject(check).optString("status"))) {
                        bundleDir = existingBundle;
                        bundleModelStateSha = readBundleModelStateSha(existingBundle);
                        append("restored .atb bundle: " + bundleModelStateSha);
                    }
                }
            } catch (Throwable t) {
                append("existing bundle restore skipped: " + t);
            }
            try {
                File existingPilot = new File(getFilesDir(), "pilot_bundle");
                if (existingPilot.isDirectory()) {
                    verifyPilotFiles(existingPilot);
                    pilotDir = existingPilot;
                    append("restored LR pilot package");
                }
            } catch (Throwable t) {
                append("existing pilot restore skipped: " + t);
            }
            try {
                File existingStage =
                        new File(getFilesDir(), "production_stage");
                if (existingStage.isDirectory()) {
                    verifyStageFiles(existingStage);
                    stageDir = existingStage;
                    append("restored production stage package");
                }
            } catch (Throwable t) {
                append("existing stage restore skipped: " + t);
            }
            try {
                File existingF2Pilot =
                        new File(getFilesDir(), "f2_sft_pilot_bundle");
                if (existingF2Pilot.isDirectory()) {
                    f2PilotSchema = verifyF2PilotFiles(existingF2Pilot);
                    f2PilotDir = existingF2Pilot;
                    append("restored F2 SFT LR pilot package");
                }
            } catch (Throwable t) {
                append("existing F2 pilot restore skipped: " + t);
            }
            try {
                File existingF2Stage =
                        new File(getFilesDir(), "f2_sft_stage_bundle");
                if (existingF2Stage.isDirectory()) {
                    f2StageSchema = verifyF2StageFiles(existingF2Stage);
                    f2StageDir = existingF2Stage;
                    append("restored F2 SFT production stage package");
                }
            } catch (Throwable t) {
                append("existing F2 stage restore skipped: " + t);
            }
            runOnUiThread(this::updateAllModeButtons);
        }, "android-trainer-restore").start();
    }

    private String readBundleModelStateSha(File root) throws Exception {
        File manifestFile = new File(root, "manifest.json");
        String text = new String(
                java.nio.file.Files.readAllBytes(manifestFile.toPath()),
                java.nio.charset.StandardCharsets.UTF_8);
        return new JSONObject(text).getString("model_state_sha256");
    }

    private void verifyBundleFiles(File root) throws Exception {
        File manifestFile = new File(root, "manifest.json");
        if (!manifestFile.isFile()) throw new IllegalStateException("manifest.json missing");
        String text = new String(java.nio.file.Files.readAllBytes(manifestFile.toPath()), java.nio.charset.StandardCharsets.UTF_8);
        JSONObject manifest = new JSONObject(text);
        if (!"android_trainer_bundle_v2".equals(manifest.getString("schema")))
            throw new IllegalStateException("wrong bundle schema");
        JSONObject tensors = manifest.getJSONObject("tensors");
        Iterator<String> names = tensors.keys();
        long total = 0;
        while (names.hasNext()) {
            String name = names.next();
            JSONObject e = tensors.getJSONObject(name);
            File f = canonicalChild(root, e.getString("path"));
            if (!f.isFile()) throw new IllegalStateException("missing tensor " + name);
            if (f.length() != e.getLong("nbytes")) throw new IllegalStateException("size mismatch " + name);
            String got = sha256(f);
            if (!got.equalsIgnoreCase(e.getString("sha256")))
                throw new IllegalStateException("SHA mismatch " + name);
            total += f.length();
        }
        File sample = canonicalChild(root, manifest.getJSONObject("sample").getString("tokens_file"));
        if (!sample.isFile() || sample.length() != 257L * 4L)
            throw new IllegalStateException("bad sample token file");
        JSONObject sampleSpec = manifest.getJSONObject("sample");
        if (!sampleSpec.has("sha256") ||
            !sha256(sample).equalsIgnoreCase(sampleSpec.getString("sha256")))
            throw new IllegalStateException("sample SHA mismatch");
        append("verified tensor bytes: " + total);
    }

    private void runGate() {
        if (bundleDir == null) return;
        append("\n=== EXACT MODEL GATE ===");
        if (Build.VERSION.SDK_INT >= 30) {
            try {
                ActivityManager am = (ActivityManager) getSystemService(ACTIVITY_SERVICE);
                if (am != null) am.setProcessStateSummary("model0001-gate-running".getBytes(java.nio.charset.StandardCharsets.UTF_8));
            } catch (Throwable ignored) {}
        }
        run.setEnabled(false);
        new Thread(() -> {
            PowerManager.WakeLock wl = null;
            try {
                wl = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "AndroidTrainer:Gate");
                wl.acquire(45L * 60L * 1000L);
                float thermal = thermalHeadroom();
                long gateStartNs = android.os.SystemClock.elapsedRealtimeNanos();
                String out = nativeRunGate(bundleDir.getAbsolutePath(), getFilesDir().getAbsolutePath(), thermal);
                float thermalEnd = thermalHeadroom();
                double gateSeconds = (android.os.SystemClock.elapsedRealtimeNanos() - gateStartNs) / 1.0e9;
                try {
                    JSONObject reportObj = new JSONObject(out);
                    reportObj.put("thermal_headroom_end", Float.isNaN(thermalEnd) ? JSONObject.NULL : thermalEnd);
                    reportObj.put("gate_wall_seconds", gateSeconds);
                    out = reportObj.toString();
                } catch (Exception ignored) {
                    // Preserve native diagnostic text if it is not JSON.
                }
                lastReport = out;
                File report = new File(getFilesDir(), "last_gate_report.json");
                try (FileOutputStream os = new FileOutputStream(report)) {
                    os.write(out.getBytes(java.nio.charset.StandardCharsets.UTF_8));
                    os.getFD().sync();
                }
                publishReport(out);
                append(pretty(out));
            } catch (Throwable t) {
                append("GATE FAIL: " + t);
            } finally {
                if (wl != null && wl.isHeld()) wl.release();
                runOnUiThread(this::updateAllModeButtons);
            }
        }, "android-trainer-gate").start();
    }


    private void runLrPilot() {
        if (bundleDir == null || pilotDir == null) return;
        append("\n=== FOUNDATION-v3 LR PILOT ===");
        if (Build.VERSION.SDK_INT >= 30) {
            try {
                ActivityManager am =
                        (ActivityManager) getSystemService(ACTIVITY_SERVICE);
                if (am != null)
                    am.setProcessStateSummary(
                            "model0001-v3-lr-pilot-running".getBytes(
                                    java.nio.charset.StandardCharsets.UTF_8));
            } catch (Throwable ignored) {}
        }
        run.setEnabled(false);
        pilotRun.setEnabled(false);
        new Thread(() -> {
            PowerManager.WakeLock wl = null;
            try {
                wl = powerManager.newWakeLock(
                        PowerManager.PARTIAL_WAKE_LOCK,
                        "AndroidTrainer:V3LrPilot");
                wl.acquire(40L * 60L * 1000L);
                float thermalStart = thermalHeadroom();
                long startNs = android.os.SystemClock.elapsedRealtimeNanos();
                String out = nativeRunLrPilot(
                        bundleDir.getAbsolutePath(),
                        pilotDir.getAbsolutePath(),
                        getFilesDir().getAbsolutePath());
                float thermalEnd = thermalHeadroom();
                double seconds =
                        (android.os.SystemClock.elapsedRealtimeNanos() - startNs)
                                / 1.0e9;
                try {
                    JSONObject reportObj = new JSONObject(out);
                    reportObj.put(
                            "thermal_headroom_start",
                            Float.isNaN(thermalStart)
                                    ? JSONObject.NULL : thermalStart);
                    reportObj.put(
                            "thermal_headroom_end",
                            Float.isNaN(thermalEnd)
                                    ? JSONObject.NULL : thermalEnd);
                    reportObj.put("pilot_wall_seconds", seconds);
                    out = reportObj.toString();
                } catch (Exception ignored) {}
                lastReport = out;
                File report = new File(
                        getFilesDir(), "last_v3_lr_pilot_report.json");
                try (FileOutputStream os = new FileOutputStream(report)) {
                    os.write(out.getBytes(
                            java.nio.charset.StandardCharsets.UTF_8));
                    os.getFD().sync();
                }
                publishReport(out);
                append(pretty(out));
            } catch (Throwable t) {
                append("LR PILOT FAIL: " + t);
            } finally {
                if (wl != null && wl.isHeld()) wl.release();
                runOnUiThread(this::updateAllModeButtons);
            }
        }, "android-trainer-v3-lr-pilot").start();
    }



    private void runF2SftLrPilot() {
        if (bundleDir == null || f2PilotDir == null ||
                !isFoundationV3Source()) return;
        append("\n=== F2 SFT ASSISTANT-ONLY LR PILOT ===");
        updateAllModeButtons();
        if (f2PilotRun != null) f2PilotRun.setEnabled(false);

        new Thread(() -> {
            PowerManager.WakeLock wl = null;
            try {
                wl = powerManager.newWakeLock(
                        PowerManager.PARTIAL_WAKE_LOCK,
                        "AndroidTrainer:F2SftPilot");
                wl.acquire(45L * 60L * 1000L);
                float thermalStart = thermalHeadroom();
                long startNs = android.os.SystemClock.elapsedRealtimeNanos();
                String out = nativeRunF2SftLrPilot(
                        bundleDir.getAbsolutePath(),
                        f2PilotDir.getAbsolutePath(),
                        getFilesDir().getAbsolutePath());
                float thermalEnd = thermalHeadroom();
                double seconds =
                        (android.os.SystemClock.elapsedRealtimeNanos() - startNs)
                                / 1.0e9;
                try {
                    JSONObject reportObj = new JSONObject(out);
                    reportObj.put(
                            "thermal_headroom_start",
                            Float.isNaN(thermalStart)
                                    ? JSONObject.NULL : thermalStart);
                    reportObj.put(
                            "thermal_headroom_end",
                            Float.isNaN(thermalEnd)
                                    ? JSONObject.NULL : thermalEnd);
                    reportObj.put("pilot_wall_seconds", seconds);
                    out = reportObj.toString();
                } catch (Exception ignored) {}

                lastReport = out;
                File report = new File(
                        getFilesDir(), "last_f2_sft_lr_pilot_report.json");
                try (FileOutputStream os = new FileOutputStream(report)) {
                    os.write(out.getBytes(
                            java.nio.charset.StandardCharsets.UTF_8));
                    os.getFD().sync();
                }
                publishReport(out);
                append(pretty(out));
            } catch (Throwable t) {
                append("F2 SFT PILOT FAIL: " + t);
            } finally {
                if (wl != null && wl.isHeld()) wl.release();
                runOnUiThread(this::updateAllModeButtons);
            }
        }, "android-trainer-f2-sft-pilot").start();
    }



    private void runF2SftStage() {
        if (bundleDir == null || f2StageDir == null ||
                !isFoundationV3Source() || trainingActive) return;
        append("\n=== MASKED SFT PRODUCTION TRAINING ===");
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        trainingActive = true;
        updateAllModeButtons();

        startF2ProgressWatcher();

        new Thread(() -> {
            PowerManager.WakeLock wl = null;
            try {
                wl = powerManager.newWakeLock(
                        PowerManager.PARTIAL_WAKE_LOCK,
                        "AndroidTrainer:F2SftProduction");
                wl.acquire(8L * 60L * 60L * 1000L);
                float thermalStart = thermalHeadroom();
                long startNs = android.os.SystemClock.elapsedRealtimeNanos();
                String out = nativeRunF2SftStage(
                        bundleDir.getAbsolutePath(),
                        f2StageDir.getAbsolutePath(),
                        getFilesDir().getAbsolutePath());
                float thermalEnd = thermalHeadroom();
                double seconds =
                        (android.os.SystemClock.elapsedRealtimeNanos() - startNs)
                                / 1.0e9;
                try {
                    JSONObject reportObj = new JSONObject(out);
                    reportObj.put(
                            "thermal_headroom_start",
                            Float.isNaN(thermalStart)
                                    ? JSONObject.NULL : thermalStart);
                    reportObj.put(
                            "thermal_headroom_end",
                            Float.isNaN(thermalEnd)
                                    ? JSONObject.NULL : thermalEnd);
                    reportObj.put("app_stage_wall_seconds", seconds);
                    JSONObject checkpoint =
                            reportObj.optJSONObject("checkpoint");
                    if (checkpoint != null) {
                        String path = checkpoint.optString("path", "");
                        if (!path.isEmpty()) {
                            File file = new File(path);
                            if (file.isFile()) lastTrainingCheckpoint = file;
                        }
                    }
                    out = reportObj.toString();
                } catch (Exception ignored) {}
                lastReport = out;
                File report =
                        new File(getFilesDir(), "last_f2_sft_stage_report.json");
                try (FileOutputStream os = new FileOutputStream(report)) {
                    os.write(out.getBytes(
                            java.nio.charset.StandardCharsets.UTF_8));
                    os.getFD().sync();
                }
                publishReport(out);
                append(pretty(out));
            } catch (Throwable t) {
                append("F2 SFT TRAINING FAIL: " + t);
            } finally {
                trainingActive = false;
                if (wl != null && wl.isHeld()) wl.release();
                runOnUiThread(() -> {
                    getWindow().clearFlags(
                            WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
                    updateAllModeButtons();
                    if (exportCheckpoint != null)
                        exportCheckpoint.setEnabled(
                                lastTrainingCheckpoint != null &&
                                lastTrainingCheckpoint.isFile());
                });
            }
        }, "android-trainer-f2-sft-production").start();
    }

    private void startF2ProgressWatcher() {
        new Thread(() -> {
            String previous = "";
            String progressName =
                    "model0001_f2r_stage_package_v1".equals(f2StageSchema)
                            ? "model0001-f2r-progress.json"
                            : "model0001-f2-sft-progress.json";
            File progress = new File(getFilesDir(), progressName);
            while (trainingActive) {
                try {
                    if (progress.isFile()) {
                        String raw = new String(
                                java.nio.file.Files.readAllBytes(progress.toPath()),
                                java.nio.charset.StandardCharsets.UTF_8);
                        if (!raw.equals(previous)) {
                            previous = raw;
                            JSONObject p = new JSONObject(raw);
                            append(String.format(
                                    Locale.ROOT,
                                    "SFT %d/%d  %.2f%%  scored=%d  lr=%.3g  loss=%s  sft=%s  v3=%s  v1=%s",
                                    p.optInt("optimizer_step", 0),
                                    p.optInt("total_updates", 0),
                                    100.0 * p.optDouble("fraction_complete", 0.0),
                                    p.optLong("scored_assistant_tokens", 0L),
                                    p.optDouble("learning_rate", Double.NaN),
                                    String.valueOf(p.opt("last_train_loss")),
                                    String.valueOf(p.opt("latest_sft_validation_ce")),
                                    String.valueOf(p.opt("latest_v3_validation_ce")),
                                    String.valueOf(p.opt("latest_v1_validation_ce"))));
                        }
                    }
                    Thread.sleep(10000L);
                } catch (InterruptedException e) {
                    return;
                } catch (Throwable t) {
                    append("F2 progress watcher: " + t);
                    try { Thread.sleep(10000L); }
                    catch (InterruptedException e) { return; }
                }
            }
        }, "android-trainer-f2-progress-watch").start();
    }


    private void runProductionStage() {
        if (bundleDir == null || stageDir == null || trainingActive) return;
        append("\n=== FOUNDATION-v3 PRODUCTION TRAINING ===");
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        trainingActive = true;
        run.setEnabled(false);
        pilotRun.setEnabled(false);
        productionRun.setEnabled(false);
        if (Build.VERSION.SDK_INT >= 30) {
            try {
                ActivityManager am =
                        (ActivityManager) getSystemService(ACTIVITY_SERVICE);
                if (am != null)
                    am.setProcessStateSummary(
                            "model0001-foundation-v3-training".getBytes(
                                    java.nio.charset.StandardCharsets.UTF_8));
            } catch (Throwable ignored) {}
        }

        startProgressWatcher();

        new Thread(() -> {
            PowerManager.WakeLock wl = null;
            try {
                wl = powerManager.newWakeLock(
                        PowerManager.PARTIAL_WAKE_LOCK,
                        "AndroidTrainer:FoundationV3");
                wl.acquire(6L * 60L * 60L * 1000L);
                float thermalStart = thermalHeadroom();
                long startNs = android.os.SystemClock.elapsedRealtimeNanos();
                String out = nativeRunStage(
                        bundleDir.getAbsolutePath(),
                        stageDir.getAbsolutePath(),
                        getFilesDir().getAbsolutePath());
                float thermalEnd = thermalHeadroom();
                double seconds =
                        (android.os.SystemClock.elapsedRealtimeNanos() - startNs)
                                / 1.0e9;
                try {
                    JSONObject reportObj = new JSONObject(out);
                    reportObj.put(
                            "thermal_headroom_start",
                            Float.isNaN(thermalStart)
                                    ? JSONObject.NULL : thermalStart);
                    reportObj.put(
                            "thermal_headroom_end",
                            Float.isNaN(thermalEnd)
                                    ? JSONObject.NULL : thermalEnd);
                    reportObj.put("app_stage_wall_seconds", seconds);
                    JSONObject checkpoint =
                            reportObj.optJSONObject("checkpoint");
                    if (checkpoint != null) {
                        String path = checkpoint.optString("path", "");
                        if (!path.isEmpty()) {
                            File file = new File(path);
                            if (file.isFile()) lastTrainingCheckpoint = file;
                        }
                    }
                    out = reportObj.toString();
                } catch (Exception ignored) {}
                lastReport = out;
                File report =
                        new File(getFilesDir(), "last_production_stage_report.json");
                try (FileOutputStream os = new FileOutputStream(report)) {
                    os.write(out.getBytes(
                            java.nio.charset.StandardCharsets.UTF_8));
                    os.getFD().sync();
                }
                publishReport(out);
                append(pretty(out));
            } catch (Throwable t) {
                append("PRODUCTION TRAINING FAIL: " + t);
            } finally {
                trainingActive = false;
                if (wl != null && wl.isHeld()) wl.release();
                runOnUiThread(() -> {
                    getWindow().clearFlags(
                            WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
                    run.setEnabled(bundleDir != null);
                    updatePilotEnabled();
                    updateProductionEnabled();
                    if (exportCheckpoint != null)
                        exportCheckpoint.setEnabled(
                                lastTrainingCheckpoint != null &&
                                lastTrainingCheckpoint.isFile());
                });
            }
        }, "android-trainer-foundation-v3").start();
    }

    private void startProgressWatcher() {
        new Thread(() -> {
            String previous = "";
            File progress =
                    new File(getFilesDir(), "model0001-production-progress.json");
            while (trainingActive) {
                try {
                    if (progress.isFile()) {
                        String raw = new String(
                                java.nio.file.Files.readAllBytes(
                                        progress.toPath()),
                                java.nio.charset.StandardCharsets.UTF_8);
                        if (!raw.equals(previous)) {
                            previous = raw;
                            JSONObject p = new JSONObject(raw);
                            int step = p.optInt("optimizer_step", 0);
                            int total = p.optInt("total_updates", 0);
                            double lr = p.optDouble("learning_rate", Double.NaN);
                            Object loss = p.opt("last_train_loss");
                            Object v3 = p.opt("latest_v3_validation_ce");
                            Object v1 = p.opt("latest_v1_validation_ce");
                            append(String.format(
                                    Locale.ROOT,
                                    "TRAIN %d/%d  %.2f%%  lr=%.3g  loss=%s  v3=%s  v1=%s",
                                    step, total,
                                    total > 0 ? 100.0 * step / total : 0.0,
                                    lr,
                                    String.valueOf(loss),
                                    String.valueOf(v3),
                                    String.valueOf(v1)));
                        }
                    }
                    Thread.sleep(10000L);
                } catch (InterruptedException e) {
                    return;
                } catch (Throwable t) {
                    append("progress watcher: " + t);
                    try { Thread.sleep(10000L); }
                    catch (InterruptedException e) { return; }
                }
            }
        }, "android-trainer-progress-watch").start();
    }

    private void exportTrainingCheckpoint() {
        if (lastTrainingCheckpoint == null ||
                !lastTrainingCheckpoint.isFile()) {
            Toast.makeText(
                    this, "No final checkpoint yet", Toast.LENGTH_SHORT).show();
            return;
        }
        Intent i = new Intent(Intent.ACTION_CREATE_DOCUMENT);
        i.addCategory(Intent.CATEGORY_OPENABLE);
        i.setType("application/octet-stream");
        i.putExtra(
                Intent.EXTRA_TITLE,
                "model0001-foundation-v3-final.atnckpt");
        startActivityForResult(i, REQ_CHECKPOINT_EXPORT);
    }

    private void writeTrainingCheckpoint(Uri uri) {
        if (lastTrainingCheckpoint == null ||
                !lastTrainingCheckpoint.isFile()) {
            append("CHECKPOINT EXPORT FAIL: no final checkpoint");
            return;
        }
        try (InputStream in =
                     new BufferedInputStream(
                             new FileInputStream(lastTrainingCheckpoint));
             OutputStream out =
                     new BufferedOutputStream(
                             getContentResolver().openOutputStream(uri, "w"))) {
            if (out == null)
                throw new IllegalStateException(
                        "null checkpoint output stream");
            copy(in, out);
            Toast.makeText(
                    this, "Checkpoint exported", Toast.LENGTH_SHORT).show();
        } catch (Throwable t) {
            append("CHECKPOINT EXPORT FAIL: " + t);
        }
    }

    private float thermalHeadroom() {
        if (Build.VERSION.SDK_INT >= 29) {
            try { return powerManager.getThermalHeadroom(0); }
            catch (Throwable ignored) {}
        }
        return Float.NaN;
    }

    private void publishReport(String report) {
        if (Build.VERSION.SDK_INT < 29) return;
        ContentValues values = new ContentValues();
        String reportPrefix = "model0001-native-opencl-gate-";
        try {
            JSONObject reportObj = new JSONObject(report);
            if ("model0001_v3_lr_pilot_report_v1".equals(
                    reportObj.optString("schema"))) {
                reportPrefix = "model0001-v3-lr-pilot-";
            } else if ("model0001_native_stage_report_v1".equals(
                    reportObj.optString("schema"))) {
                reportPrefix = "model0001-foundation-v3-stage-";
            } else if ("model0001_f2_sft_lr_pilot_report_v1".equals(
                    reportObj.optString("schema"))) {
                reportPrefix = "model0001-f2-sft-lr-pilot-";
            } else if ("model0001_f2_sft_stage_report_v1".equals(
                    reportObj.optString("schema"))) {
                reportPrefix = "model0001-f2-sft-stage-";
            } else if ("model0001_f2r_lr_pilot_report_v1".equals(
                    reportObj.optString("schema"))) {
                reportPrefix = "model0001-f2r-lr-pilot-";
            } else if ("model0001_f2r_stage_report_v1".equals(
                    reportObj.optString("schema"))) {
                reportPrefix = "model0001-f2r-stage-";
            }
        } catch (Exception ignored) {}
        values.put(MediaStore.MediaColumns.DISPLAY_NAME,
                reportPrefix + System.currentTimeMillis() + ".json");
        values.put(MediaStore.MediaColumns.MIME_TYPE, "application/json");
        values.put(MediaStore.MediaColumns.RELATIVE_PATH,
                Environment.DIRECTORY_DOWNLOADS + "/Android-Trainer");
        values.put(MediaStore.MediaColumns.IS_PENDING, 1);
        Uri uri = getContentResolver().insert(
                MediaStore.Downloads.EXTERNAL_CONTENT_URI, values);
        if (uri == null) throw new IllegalStateException("cannot create public report");
        try (OutputStream stream = getContentResolver().openOutputStream(uri, "w")) {
            if (stream == null) throw new IllegalStateException("null public report stream");
            stream.write(report.getBytes(java.nio.charset.StandardCharsets.UTF_8));
        } catch (Exception e) {
            getContentResolver().delete(uri, null, null);
            throw new IllegalStateException("public report write failed", e);
        }
        ContentValues ready = new ContentValues();
        ready.put(MediaStore.MediaColumns.IS_PENDING, 0);
        getContentResolver().update(uri, ready, null, null);
        append("public_report_uri: " + uri);
    }

    private void exportReport() {
        if (lastReport.isEmpty()) {
            Toast.makeText(this, "No report yet", Toast.LENGTH_SHORT).show();
            return;
        }
        Intent i = new Intent(Intent.ACTION_CREATE_DOCUMENT);
        i.addCategory(Intent.CATEGORY_OPENABLE);
        i.setType("application/json");
        String title = "model0001-gpu-gate-report.json";
        try {
            JSONObject reportObj = new JSONObject(lastReport);
            if ("model0001_v3_lr_pilot_report_v1".equals(
                    reportObj.optString("schema"))) {
                title = "model0001-v3-lr-pilot-report.json";
            } else if ("model0001_native_stage_report_v1".equals(
                    reportObj.optString("schema"))) {
                title = "model0001-foundation-v3-stage-report.json";
            } else if ("model0001_f2_sft_lr_pilot_report_v1".equals(
                    reportObj.optString("schema"))) {
                title = "model0001-f2-sft-lr-pilot-report.json";
            } else if ("model0001_f2_sft_stage_report_v1".equals(
                    reportObj.optString("schema"))) {
                title = "model0001-f2-sft-stage-report.json";
            } else if ("model0001_f2r_lr_pilot_report_v1".equals(
                    reportObj.optString("schema"))) {
                title = "model0001-f2r-lr-pilot-report.json";
            } else if ("model0001_f2r_stage_report_v1".equals(
                    reportObj.optString("schema"))) {
                title = "model0001-f2r-stage-report.json";
            }
        } catch (Exception ignored) {}
        i.putExtra(Intent.EXTRA_TITLE, title);
        startActivityForResult(i, REQ_REPORT);
    }

    private void writeReport(Uri uri) {
        try (OutputStream out = getContentResolver().openOutputStream(uri, "wt")) {
            if (out == null) throw new IllegalStateException("null report stream");
            out.write(lastReport.getBytes(java.nio.charset.StandardCharsets.UTF_8));
            Toast.makeText(this, "Report exported", Toast.LENGTH_SHORT).show();
        } catch (Throwable t) {
            append("REPORT EXPORT FAIL: " + t);
        }
    }

    private void exportNativeTrace() {
        if (lastNativeTrace == null || !lastNativeTrace.isFile()) {
            Toast.makeText(this, "No native tombstone captured", Toast.LENGTH_SHORT).show();
            return;
        }
        Intent i = new Intent(Intent.ACTION_CREATE_DOCUMENT);
        i.addCategory(Intent.CATEGORY_OPENABLE);
        i.setType("application/octet-stream");
        i.putExtra(Intent.EXTRA_TITLE, lastNativeTrace.getName());
        startActivityForResult(i, REQ_CRASH_TRACE);
    }

    private void writeNativeTrace(Uri uri) {
        if (lastNativeTrace == null || !lastNativeTrace.isFile()) {
            append("CRASH TRACE EXPORT FAIL: no captured tombstone");
            return;
        }
        try (InputStream in = new BufferedInputStream(new FileInputStream(lastNativeTrace));
             OutputStream out = getContentResolver().openOutputStream(uri, "w")) {
            if (out == null) throw new IllegalStateException("null crash trace stream");
            copy(in, out);
            Toast.makeText(this, "Native crash trace exported", Toast.LENGTH_SHORT).show();
        } catch (Throwable t) {
            append("CRASH TRACE EXPORT FAIL: " + t);
        }
    }

    private static void copy(InputStream in, OutputStream out) throws Exception {
        byte[] buf = new byte[1 << 16];
        int n;
        while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
    }

    private static File canonicalChild(File root, String rel) throws Exception {
        File f = new File(root, rel).getCanonicalFile();
        String rp = root.getCanonicalPath() + File.separator;
        if (!f.getPath().startsWith(rp)) throw new SecurityException("path traversal: " + rel);
        return f;
    }

    private static void unzipSafely(File zip, File root) throws Exception {
        try (ZipInputStream zin = new ZipInputStream(new BufferedInputStream(new FileInputStream(zip)))) {
            ZipEntry e;
            while ((e = zin.getNextEntry()) != null) {
                if (e.isDirectory()) continue;
                File dst = canonicalChild(root, e.getName());
                File parent = dst.getParentFile();
                if (!parent.isDirectory() && !parent.mkdirs()) throw new IllegalStateException("mkdir failed");
                try (OutputStream out = new BufferedOutputStream(new FileOutputStream(dst))) { copy(zin, out); }
                zin.closeEntry();
            }
        }
    }

    private static void deleteTree(File f) {
        if (f == null || !f.exists()) return;
        if (f.isDirectory()) {
            File[] xs = f.listFiles();
            if (xs != null) for (File x : xs) deleteTree(x);
        }
        if (!f.delete()) { /* best effort; following mkdir/write will surface a real failure */ }
    }

    private static String sha256(File f) throws Exception {
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        try (InputStream in = new BufferedInputStream(new FileInputStream(f))) {
            byte[] buf = new byte[1 << 16];
            int n;
            while ((n = in.read(buf)) > 0) md.update(buf, 0, n);
        }
        StringBuilder sb = new StringBuilder();
        for (byte b : md.digest()) sb.append(String.format(Locale.ROOT, "%02x", b & 0xff));
        return sb.toString();
    }
}
