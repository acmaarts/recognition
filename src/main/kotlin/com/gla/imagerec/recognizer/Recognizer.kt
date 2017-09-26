package com.jacksierkstra.tensorflow

import com.gla.imagerec.recognizer.Recognized
import org.tensorflow.*

import java.io.FileNotFoundException
import java.io.IOException
import java.net.URL
import java.nio.charset.Charset
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.util.ArrayList
import java.util.Arrays

class Recognizer : Recognizable {

    private var graphDef: ByteArray? = null
    private var labels: List<String>? = ArrayList()

    init {

    }

    @Throws(Exception::class)
    override fun recognize(imageBytes: ByteArray) : Recognized {

        constructAndExecuteGraphToNormalizeImage(imageBytes).use { image ->
            val labelProbabilities = executeInceptionGraph(graphDef!!, image)
            val bestLabelIdx = maxIndex(labelProbabilities)
            return Recognized(labels?.get(bestLabelIdx),labelProbabilities[bestLabelIdx] * 100f)
        }

    }

    @Throws(Exception::class)
    override fun recognize(file: String) {

        if (graphDef == null) {
            println("No graph loaded.")
            return
        }

        val imageBytes = readAllBytesOrExit(getPath(file))!!

        constructAndExecuteGraphToNormalizeImage(imageBytes).use { image ->
            val labelProbabilities = executeInceptionGraph(graphDef!!, image)
            val bestLabelIdx = maxIndex(labelProbabilities)
            println(
                    String.format(
                            "BEST MATCH: %s (%.2f%% likely)",
                            labels?.get(bestLabelIdx), labelProbabilities[bestLabelIdx] * 100f))
        }

    }

    @Throws(Exception::class)
    override fun loadGraph(file: String) {
        graphDef = readAllBytesOrExit(getPath(file))
    }

    @Throws(Exception::class)
    override fun loadLabels(file: String) {

        labels = readAllLinesOrExit(getPath(file))
    }

    @Throws(Exception::class)
    private fun getPath(file: String): Path {
        var path: Path? = null
        val resource = this.javaClass.classLoader.getResource(file)

        if (resource != null) {
            path = Paths.get(resource.path)
        }

        if (path == null) {
            throw FileNotFoundException(String.format("File with filename: %s could not be found.", file))
        }

        return path
    }

    private fun constructAndExecuteGraphToNormalizeImage(imageBytes: ByteArray): Tensor {
        Graph().use { g ->
            val b = GraphBuilder(g)
            // Some constants specific to the pre-trained model at:
            // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
            //
            // - The model was trained with images scaled to 224x224 pixels.
            // - The colors, represented as R, G, B in 1-byte each were converted to
            //   float using (value - Mean)/Scale.
            val H = 224
            val W = 224
            val mean = 117f
            val scale = 1f

            // Since the graph is being constructed once per execution here, we can use a constant for the
            // input image. If the graph were to be re-used for multiple input images, a placeholder would
            // have been more appropriate.
            val input = b.constant("input", imageBytes)
            val output = b.div(
                    b.sub(
                            b.resizeBilinear(
                                    b.expandDims(
                                            b.cast(b.decodeJpeg(input, 3), DataType.FLOAT),
                                            b.constant("make_batch", 0)),
                                    b.constant("size", intArrayOf(H, W))),
                            b.constant("mean", mean)),
                    b.constant("scale", scale))
            Session(g).use { s -> return s.runner().fetch(output.op().name()).run()[0] }
        }
    }

    private fun executeInceptionGraph(graphDef: ByteArray, image: Tensor): FloatArray {
        Graph().use { g ->
            g.importGraphDef(graphDef)
            Session(g).use { s ->
                s.runner().feed("input", image).fetch("output").run()[0].use { result ->
                    val rshape = result.shape()
                    if (result.numDimensions() != 2 || rshape[0] != 1L) {
                        throw RuntimeException(
                                String.format(
                                        "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                                        Arrays.toString(rshape)))
                    }
                    val nlabels = rshape[1].toInt()
                    return result.copyTo(Array(1) { FloatArray(nlabels) })[0]
                }
            }
        }
    }

    private fun maxIndex(probabilities: FloatArray): Int {
        var best = 0
        for (i in 1..probabilities.size - 1) {
            if (probabilities[i] > probabilities[best]) {
                best = i
            }
        }
        return best
    }

    private fun readAllBytesOrExit(path: Path): ByteArray? {
        try {
            return Files.readAllBytes(path)
        } catch (e: IOException) {
            System.err.println("Failed to read [" + path + "]: " + e.message)
            System.exit(1)
        }

        return null
    }

    private fun readAllLinesOrExit(path: Path): List<String>? {
        try {
            return Files.readAllLines(path, Charset.forName("UTF-8"))
        } catch (e: IOException) {
            System.err.println("Failed to read [" + path + "]: " + e.message)
            System.exit(0)
        }

        return null
    }

    // In the fullness of time, equivalents of the methods of this class should be auto-generated from
    // the OpDefs linked into libtensorflow_jni.so. That would match what is done in other languages
    // like Python, C++ and Go.
    internal class GraphBuilder(private val g: Graph) {

        fun div(x: Output, y: Output): Output {
            return binaryOp("Div", x, y)
        }

        fun sub(x: Output, y: Output): Output {
            return binaryOp("Sub", x, y)
        }

        fun resizeBilinear(images: Output, size: Output): Output {
            return binaryOp("ResizeBilinear", images, size)
        }

        fun expandDims(input: Output, dim: Output): Output {
            return binaryOp("ExpandDims", input, dim)
        }

        fun cast(value: Output, dtype: DataType): Output {
            return g.opBuilder("Cast", "Cast").addInput(value).setAttr("DstT", dtype).build().output(0)
        }

        fun decodeJpeg(contents: Output, channels: Long): Output {
            return g.opBuilder("DecodeJpeg", "DecodeJpeg")
                    .addInput(contents)
                    .setAttr("channels", channels)
                    .build()
                    .output(0)
        }

        fun constant(name: String, value: Any): Output {
            Tensor.create(value).use { t ->
                return g.opBuilder("Const", name)
                        .setAttr("dtype", t.dataType())
                        .setAttr("value", t)
                        .build()
                        .output(0)
            }
        }

        private fun binaryOp(type: String, in1: Output, in2: Output): Output {
            return g.opBuilder(type, type).addInput(in1).addInput(in2).build().output(0)
        }
    }

}
