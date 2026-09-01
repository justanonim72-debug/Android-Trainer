package dev.riszn.androidtrainer;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.PowerManager;
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
import java.util.Locale;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public final class MainActivity extends Activity {
    private static final int REQ_BUNDLE = 1001;
    private static final int REQ_REPORT = 1002;

    static {
        System.loadLibrary("android_trainer");
    }

    private TextView log;
    private Button run;
    private File bundleDir;
    private String lastReport = "";
    private PowerManager powerManager;

    private static native String nativeProbe();
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
        subtitle.setText("FP32 correctness → real backward → AdamW → sustained speed");
        subtitle.setTextColor(0xff9aa0a6);
        subtitle.setTextSize(13);
        subtitle.setPadding(0, 6, 0, 20);
        root.addView(subtitle, new LinearLayout.LayoutParams(-1, -2));

        LinearLayout buttons = new LinearLayout(this);
        buttons.setOrientation(LinearLayout.VERTICAL);

        Button probe = button("1. Probe device + backends");
        probe.setOnClickListener(v -> background("PROBE", () -> nativeProbe()));
        buttons.addView(probe);

        Button select = button("2. Import local .atb bundle");
        select.setOnClickListener(v -> selectBundle());
        buttons.addView(select);

        run = button("3. Run exact GPU Gate");
        run.setEnabled(false);
        run.setOnClickListener(v -> runGate());
        buttons.addView(run);

        Button export = button("Export last JSON report");
        export.setOnClickListener(v -> exportReport());
        buttons.addView(export);

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
    }

    private void importBundle(Uri uri) {
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
                bundleDir = dest;
                runOnUiThread(() -> run.setEnabled(true));
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
        run.setEnabled(false);
        new Thread(() -> {
            PowerManager.WakeLock wl = null;
            try {
                wl = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "AndroidTrainer:Gate");
                wl.acquire(45L * 60L * 1000L);
                float thermal = thermalHeadroom();
                String out = nativeRunGate(bundleDir.getAbsolutePath(), getFilesDir().getAbsolutePath(), thermal);
                lastReport = out;
                File report = new File(getFilesDir(), "last_gate_report.json");
                try (FileOutputStream os = new FileOutputStream(report)) {
                    os.write(out.getBytes(java.nio.charset.StandardCharsets.UTF_8));
                    os.getFD().sync();
                }
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
