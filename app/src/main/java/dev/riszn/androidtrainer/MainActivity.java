package dev.riszn.androidtrainer;

import android.app.Activity;
import android.app.ActivityManager;
import android.app.ApplicationExitInfo;
import android.content.ContentValues;
import android.content.Intent;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.PowerManager;
import android.provider.MediaStore;
import android.provider.Settings;
import android.view.Gravity;
import android.view.View;
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

    static {
        System.loadLibrary("android_trainer");
    }

    private TextView log;
    private Button run;
    private File bundleDir;
    private String lastReport = "";
    private File lastNativeTrace;
    private PowerManager powerManager;

    private static native String nativeProbe();
    private static native String nativeOpenClConformance();
    private static native String nativeValidateBundle(String bundleDir);
    private static native String nativeRunGate(String bundleDir, String workDir, float thermalHeadroom);

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        powerManager = (PowerManager) getSystemService(POWER_SERVICE);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(28, 28, 28, 28);
        root.setBackgroundColor(0xff101114);

        TextView title = new TextView(this);
        title.setText("MODEL #0001  •  GPU GATE");
        title.setTextColor(0xfff1f3f4);
        title.setTextSize(20);
        title.setGravity(Gravity.CENTER_VERTICAL);
        root.addView(title, new LinearLayout.LayoutParams(-1, -2));

        TextView subtitle = new TextView(this);
        subtitle.setText("Pure OpenCL C 1.2 FP32 buffer conformance • MNN GPU retired");
        subtitle.setTextColor(0xff9aa0a6);
        subtitle.setTextSize(13);
        subtitle.setPadding(0, 6, 0, 20);
        root.addView(subtitle, new LinearLayout.LayoutParams(-1, -2));

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

        Button export = button("Export last JSON report");
        export.setOnClickListener(v -> exportReport());
        buttons.addView(export);

        Button exportCrash = button("Export native crash trace");
        exportCrash.setOnClickListener(v -> exportNativeTrace());
        buttons.addView(exportCrash);

        root.addView(buttons, new LinearLayout.LayoutParams(-1, -2));

        log = new TextView(this);
        log.setTextColor(0xffd7dadc);
        log.setTextSize(12);
        log.setTextIsSelectable(true);
        log.setTypeface(android.graphics.Typeface.MONOSPACE);
        log.setPadding(0, 20, 0, 40);
        log.setText("No bundle loaded.\n");

        ScrollView scroller = new ScrollView(this);
        scroller.addView(log);
        root.addView(scroller, new LinearLayout.LayoutParams(-1, 0, 1f));
        setContentView(root);
        showPreviousExitDiagnostics();
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

    private Button button(String text) {
        Button b = new Button(this);
        b.setText(text);
        b.setAllCaps(false);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(-1, -2);
        lp.setMargins(0, 5, 0, 5);
        b.setLayoutParams(lp);
        return b;
    }

    private void append(String s) {
        runOnUiThread(() -> log.append(s + "\n"));
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
        if (requestCode == REQ_REPORT) writeReport(data.getData());
        if (requestCode == REQ_CRASH_TRACE) writeNativeTrace(data.getData());
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
                runOnUiThread(() -> {
                    run.setEnabled(true);
                    if (runAfterImport) runGate();
                });
            } catch (Throwable t) {
                append("IMPORT FAIL: " + t);
            }
        }, "android-trainer-import").start();
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
                runOnUiThread(() -> run.setEnabled(bundleDir != null));
            }
        }, "android-trainer-gate").start();
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
        values.put(MediaStore.MediaColumns.DISPLAY_NAME,
                "model0001-native-opencl-gate-" + System.currentTimeMillis() + ".json");
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
        i.putExtra(Intent.EXTRA_TITLE, "model0001-gpu-gate-report.json");
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
