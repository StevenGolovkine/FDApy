#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the class FCPTPA in the fcp_tpa.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from FDApy.representation.functional_data import DenseFunctionalData
from FDApy.simulation.karhunen import KarhunenLoeve
from FDApy.misc.utils import _eigh
from FDApy.preprocessing.dim_reduction.fcp_tpa import (
    FCPTPA,
    _initialize_vectors,
    _initalize_output,
    _eigendecomposition_penalty_matrices,
    _gcv,
    _find_optimal_alpha,
    _compute_denominator,
    _update_vector,
    _update_components
)
from numpy.linalg import norm


class TestInitializeVectors(unittest.TestCase):
    def test_initialize_vectors_shape(self):
        shape = (5, 10, 15)
        u, v, w = _initialize_vectors(shape)

        # Check that the shape of each vector is correct
        np.testing.assert_equal(u.shape, (5,))
        np.testing.assert_equal(v.shape, (10,))
        np.testing.assert_equal(w.shape, (15,))

    def test_initialize_vectors_norm(self):
        shape = (5, 10, 15)
        u, v, w = _initialize_vectors(shape)

        # Check that the norm of each vector is approximately 1
        np.testing.assert_almost_equal(norm(u), 1)
        np.testing.assert_almost_equal(norm(v), 1)
        np.testing.assert_almost_equal(norm(w), 1)

    def test_initialize_vectors_range(self):
        shape = (5, 10, 15)
        u, v, w = _initialize_vectors(shape)

        # Check that each element of the vectors is between -1 and 1
        np.testing.assert_array_less(np.abs(u), 1)
        np.testing.assert_array_less(np.abs(v), 1)
        np.testing.assert_array_less(np.abs(w), 1)


class TestInitializeOutput(unittest.TestCase):
    def test_initialize_output(self):
        shape = (5, 10, 15)
        n_components = 3
        output = _initalize_output(shape, n_components)

        # Check the shape of the matrices
        np.testing.assert_equal(output[0].shape, (n_components,))
        np.testing.assert_equal(output[1][0].shape, (shape[0], n_components))
        np.testing.assert_equal(output[1][1].shape, (shape[1], n_components))
        np.testing.assert_equal(output[1][2].shape, (shape[2], n_components))

        # Check that the matrices are initialized to zeros
        np.testing.assert_array_equal(output[0], np.zeros(n_components))
        np.testing.assert_array_equal(output[1][0], np.zeros((shape[0], n_components)))
        np.testing.assert_array_equal(output[1][1], np.zeros((shape[1], n_components)))
        np.testing.assert_array_equal(output[1][2], np.zeros((shape[2], n_components)))


class TestEigenDecompositionPenaltyMatrices(unittest.TestCase):
    def test_eigendecomposition_penalty_matrices(self):
        mat_v = np.diff(np.identity(3))
        mat_w = np.diff(np.identity(5))
        penal_mat = {'v': np.dot(mat_v, mat_v.T), 'w': np.dot(mat_w, mat_w.T)}

        eigen = _eigendecomposition_penalty_matrices(penal_mat)

        expected_val_v = np.array(
            [3.00000000e+00, 1.00000000e+00, 2.56090598e-16]
        )
        expected_vec_v = np.array([
            [4.08248290e-01, 7.07106781e-01, 5.77350269e-01],
            [8.16496581e-01, 3.75646961e-17, 5.77350269e-01],
            [4.08248290e-01, 7.07106781e-01, 5.77350269e-01]
        ])
        expected_val_w = np.array([3.61803399e+00, 2.61803399e+00, 1.38196601e+00, 3.81966011e-01, -4.08055791e-17])
        expected_vec_w = np.array([
            [ 1.95439508e-01, 3.71748034e-01, 5.11667274e-01, 6.01500955e-01, 4.47213595e-01],
            [ 5.11667274e-01, 6.01500955e-01, 1.95439508e-01, 3.71748034e-01, 4.47213595e-01],
            [ 6.32455532e-01, 1.84484383e-16, 6.32455532e-01, 1.65146213e-16, 4.47213595e-01],
            [ 5.11667274e-01, 6.01500955e-01, 1.95439508e-01, 3.71748034e-01, 4.47213595e-01],
            [ 1.95439508e-01, 3.71748034e-01, 5.11667274e-01, 6.01500955e-01, 4.47213595e-01]
        ])

        np.testing.assert_array_almost_equal(eigen['v'][0], expected_val_v)
        np.testing.assert_array_almost_equal(np.abs(eigen['v'][1]), expected_vec_v)
        np.testing.assert_array_almost_equal(eigen['w'][0], expected_val_w)
        np.testing.assert_array_almost_equal(np.abs(eigen['w'][1]), expected_vec_w)


class TestGCV(unittest.TestCase):
    def setUp(self):
        self.alpha = 2
        self.vector = np.array([
            0.0128771, -0.18092739, -0.04849094, 0.03192246, -0.01771499,
            -0.00348176, -0.05193259, 0.00363752, -0.03085957, 0.04653336,
            -0.15876299, -0.02367582, 0.10633316, -0.13322367, -0.06599965,
            -0.13597547, 0.08641482, 0.00676493, -0.03432828, 0.03044963
        ])
        self.smoother = 1.531051200361189
        self.rayleigh = np.array([
            -0.66788406, -1.36231223, -1.03999022, 0.95202651, -0.21907907,
            2.08333614, 1.64407371, -0.2898391 , 0.20132169, -0.33065561,
            -1.24685903, -2.72341886, -1.02037365, 1.58162608, 0.20662958,
            0.67221107, 0.57921458, -0.25418882, 1.0360842 , 0.18984492
        ])

    def test_gcv(self):
        expected_output = 0.08590502006053882
        output = _gcv(
            self.alpha, len(self.rayleigh),
            self.vector, self.smoother, self.rayleigh
        )
        np.testing.assert_almost_equal(output, expected_output)


class TestFindOptimalAlpha(unittest.TestCase):
    def setUp(self):
        a1 = np.linspace(0, 1, 10)
        a2 = np.linspace(-1, 1, 20)
        a3 = np.linspace(-np.pi, np.pi, 15)

        self.data = np.outer(np.outer(a1, a2), a3).reshape((10, 20, 15))
        self.alpha_range = (1e-4, 1e4)

        self.u = np.array([0.07137173, 1.75237922, -0.03798677, -2.50124779, -1.43330569, 1.04591596, -0.52735294, 1.04735957, 1.34477470, -0.73625421])
        self.v = np.array([0.359870553829616,-0.0301719673257529,-0.868129058143359,0.642072021563352,1.19301299885825,-0.0628880563553948,-1.05779116563454,0.100391706721132,-1.43084375256215,-0.133080519009047,1.83360532082849,-0.0466912290126376,-1.13008413617535,-0.37070456119696,-0.909674663612033,-0.1527568934256,-2.05035280306506,-1.32269793405163,0.933468124387712,-0.344920804168726])
        self.w = np.array([-0.600791900317172,-0.26409280629362,0.855391031790754, 0.687981300024229,-0.284703726003988,-2.0049301467036, -0.458949187552899,-0.0996207614010958,-1.02741326488901,0.991242524135272,-0.757537348261641,0.582689846436163,-0.551318737404864,0.0164132148649893,-0.892520541054338])
        self.alpha = 0.5
        self.penalty_matrix = np.array([
            [0.970366323996227,-0.573651961557894,0.54534048456886,-0.256246708715215,1.55586920388696,-0.490150118527004,0.428543605640612,0.278599231745152,-0.676154240765476,-0.0692537244943682,-0.0244353386644789,-0.474501606355632,-0.434319689545916,0.383118199013127,0.988066948047288],
            [0.225728882472099,0.972464743730131,-0.0806152097558943,0.988275968880429,-1.00620899802858,2.12947337532374,-2.21729331176844,0.8702848517818,0.0345116100898053,-0.47076776203276,1.29028222035847,-0.528118564110436,-0.640627376258532,0.0614224489719385,-0.643553511083002],
            [-0.971828413352645,0.629045249758617,-0.99185841886388,-0.433323131881468,1.75139681509155,-0.362521833982692,0.335867221380579,-0.60466995284903,-0.0328542338409614,-0.109355799783997,1.64814595477599,-1.38736417645832,-0.0589864970116763,0.883867428334307,1.04366367937375],
            [-0.319267622806463,-1.56122013695982,0.246827936352083,0.901851039728005,-0.509087467236602,0.724932694392776,-0.899481626750302,-0.563716250923417,0.268621779528051,-0.941832641800219,-2.45276453591221,0.959227427815175,0.958496586835225,0.334107348891202,0.940177335390862],
            [-0.46805481034751,-0.902177676742073,-1.56076495344266,1.55726898950104,0.0472758634977803,0.0344235234151197,0.349238429666161,-0.625063409815799,-0.624729554381001,0.124903223831373,0.214024124852007,-2.58446815029991,-2.37526946193962,2.63284882489555,0.160239672027078],
            [0.885205633992677,1.31484732774891,1.31616348618266,-0.260185610426709,0.102099410666946,-0.8495393588825,-0.172812414197505,-0.00472058615753763,-0.364859067879463,0.155328809871547,0.844468311409598,-0.915152919226084,1.76490069658829,0.126698237602997,1.02870513836732],
            [1.26714157964797,0.0913663993771958,-1.45779216720389,0.148950356714107,-1.66554318106184,-0.0484608382280268,0.490209170294078,0.606059002530827,-1.51323743988411,-0.451688672991209,-0.277778589956854,0.384655200755514,-0.641499068126161,1.62337135036539,1.32503694376387],
            [0.00318645999796357,0.515336692610761,0.487395230645122,0.833942818950253,-2.6462657143282,-0.228851652761235,1.10993335358505,-1.85067210291042,1.5144476636699,0.481949514649733,-0.408220533269312,-2.19758556618082,-0.253095220085035,0.0828222998837502,1.09461840491059],
            [-0.161701260798577,0.495116996779144,0.741135052431477,0.496605561773435,-0.00903649295146409,0.925417621984007,-1.25017960581841,0.893579327908039,-2.01829452258945,-1.34463978981481,0.655499088144141,-0.815174055201978,-1.14101546614622,0.312878893317481,-0.351184817435087],
            [-0.177415363389402,-1.55540587019317,0.168967623526794,-0.901916430345041,-1.91170516255175,-1.53721782370261,-0.187576415808032,0.0309356635423938,-0.74149639698296,0.368660850995952,-0.705988189274127,0.529164911805788,-0.520629274104655,-0.434998323113036,1.05859028918549],
            [-1.25231168687617,1.73397493584275,0.376390389524293,-1.52465428810672,-1.40985607751138,0.950861825140464,-0.285003192258542,1.56020982574499,-1.75552097408039,0.0707873090667357,0.0893176654541613,-0.109180708157059,-0.135083223651774,0.311521812980403,-0.439728945021666],
            [0.0603759386366196,-0.765441255900933,-0.0943028926168314,-1.65489090075834,1.23208068351047,0.944814974872194,0.463953708465961,0.303557291860249,2.04681582287049,-0.00221169870393254,0.67935586986908,-0.382925371965074,-1.05525915943341,0.565008455769331,0.54684132777381],
            [-1.43596292692485,-1.15080641432578,0.420404988716717,-1.04691447178835,0.327963140302636,-1.06520944023482,-0.507071715790125,-0.949110197849589,-0.436179411858169,-1.30867780421196,0.81335104602313,1.66766972000342,0.965119842308412,0.479114410477252,1.77239955524383],
            [0.61275949715169,-1.07591994493125,1.09303372550294,0.208474283634887,0.780893676671352,-0.17839993620128,0.0100678499624518,0.836302113352479,-0.821013097814598,1.57627957093056,-0.335635813987157,1.0852709798516,-0.014159028231758,0.0055168704610009,0.218575218983505],
            [1.56993148921946,-1.0029997522784,0.730556037853379,-0.0337926235121439,1.08300188500855,0.220201412200722,-1.68266522676443,-0.182148000651603,0.212739546968112,0.935174572597659,-1.24576005601009,2.90902271737388,-0.667708471081367,-1.43788011265583,0.454257603035076]
        ])
        self.eigenvectors = np.array([
            [1.53221782484906,-1.43416783305486,-0.0864047247493963,1.94413730506052,-0.000541488673555216,-1.04685137343891,-1.80857988779578,-0.96184972792885,-0.661504051719223,0.973675872933788,-0.761741931382519,-0.327822063106168,1.76889895932627,-0.242154347729835,-1.19525589562916,-0.316571392846976,0.510175454229805,0.5462320465046,-1.73486251382461,0.799476886627882],
            [-0.129759780616901,0.679153215942362,0.774178402864321,1.70425641653578,1.88819971594054,0.888717141825095,-0.0383986781199478,-1.27773992620194,0.688299021883307,-0.266911999121487,1.21089034708846,2.12327115557847,-0.68412989824739,-1.06265853683072,-1.08542839381024,-0.974688519185366,-0.828143922616152,0.721863447747789,-1.31155584483259,0.515276286664565],
            [-0.0808333102381047,-1.35263779088023,0.736362990817583,-0.939850420126451,0.887777586549804,0.806589382014066,-0.771313373035328,1.40884981676845,0.40563479831081,0.443854462708655,0.78965926665337,-0.522461914977752,1.30778191680749,-2.09746004405563,-1.26583948350561,-0.0488443472165996,0.0906926847156143,-0.306741864899404,-0.832865460411368,0.32796530833376],
            [0.358262596121188,-0.886986580561616,-1.1574803215953,-1.92967830646813,-0.755031584793634,0.791646970416564,-0.394340450625394,-0.637699391215147,-0.710340479821465,-0.196883055957865,-0.352277551588435,-1.22562340169655,0.273959083401947,-1.28129239743392,1.27154062816383,0.306387588136132,0.628883927267702,-0.129590060803633,1.27260954883526,0.0871667201310359],
            [1.31475979213307,-1.73873734859078,0.824874045033971,-0.322249134839778,-1.55389595368408,-1.03010261901389,1.63105523602603,-1.14128219639066,-0.0524076078137429,-1.43141818095021,0.493869686183257,-2.62775606241904,0.285745850872284,-0.0339766276719701,1.25772394847451,-1.32667124987422,-0.0775588326741404,-1.05283163434506,1.57157748380669,-0.35611321877649],
            [-0.0486206895135859,-0.260500793082027,0.120562180729644,0.626111603670847,-0.663188670580526,-0.848160979793841,-1.72659322475312,0.833988669482055,-0.920645088630937,1.47248323048965,-2.30185890439673,-0.15725805061417,-0.599730735787811,-0.0990856421933454,-0.620047970693506,-1.50356653058236,0.153754296517789,-1.15477604884385,1.11103357419403,0.342320197509883],
            [0.128030871315454,-0.872951806309595,0.982345845034556,-0.416238389273564,-0.453244397131295,-1.45288212771587,-0.768992150497795,2.49032561467859,0.759100343459997,2.43628812417949,-0.770092337098188,1.43111269092819,0.633691241270695,-1.45419942747413,0.869253245337832,1.83096942490759,2.60081796310241,1.25071793449869,0.0113872822643266,-1.23532216196817],
            [0.915392462084641,-0.411171797666012,-0.285617094119516,2.33519298579075,1.88954070334068,-1.06533838164665,-2.04491126814409,-1.16949125963398,1.22551137606234,-0.45684992596157,-0.984015779316577,0.356522269291416,3.15549950430293,1.1846205008926,1.16205081407215,1.48951346104164,2.05882452307825,0.758431055486023,1.71893749954605,-0.137227741702672],
            [-0.567805587090298,-1.12451750426145,-0.425186545495691,-0.489888143081571,0.994854769331176,-0.4519150994408,1.2721335330576,-0.00570342660209135,-0.183041708750762,-0.729850580109482,-1.25696322280451,0.296618276272202,-1.25707785070536,0.0268394924620557,-0.0986231836447885,0.330806550588642,-1.13472994229942,0.647888428434885,-0.694410238891795,0.427797735639959],
            [-0.0795340180931833,-1.42369887081878,-0.282645635067009,-0.495785897324464,-1.00745066597174,-0.82696776839382,-0.843193490234284,1.48271926933258,-0.287206023702426,1.01743464095271,-1.07542073790032,-0.544909023258788,-0.566665262479135,0.00873704078058586,-0.0413165905473249,-0.376886655971042,-2.34609384238704,-1.39859696483356,0.0551022861067554,-0.544887402230248],
            [-1.42393027378814,-0.0822012767017553,0.366894223596896,-0.727418243738541,-1.05796493354948,-0.663873980534471,0.0143638894034777,0.116640946815941,-0.402564057249469,0.144251058195749,0.591687042637889,0.900967489700636,0.432558483658768,2.27438161404471,0.233903577735206,-0.431118820857391,-1.42360529438805,1.46418391409145,-0.144244972979594,-0.255116280917633],
            [0.279677603620455,-0.419120731089236,-0.740006392910666,-0.679256028635527,-0.334877909286102,-0.357608110659119,-0.0316850598102859,-1.69813705462462,0.651144216211236,0.496620924187662,-1.3735189958407,1.00047536459603,-1.50354010523485,-0.80934052409521,-0.0930846275838987,-0.466142032907731,0.60780174528935,1.45472762436713,-0.0711688951699351,0.57484444181533],
            [-1.32364240321185,0.206019846022254,-1.0394645974152,0.601623801089065,-0.487756744432777,0.715942147985019,-1.88628156853189,-1.28927101432693,0.518986246723804,1.49467578610799,-0.797771730396765,-0.437385640339814,-0.047989862631524,-0.327970185780848,1.34437584966889,0.381564041544673,0.0670335357492001,-1.03697858317616,0.370175881225531,2.29151077697638],
            [0.309266787541008,1.0384778949055,1.34112674181344,-0.518180506963463,-0.912545343212438,-0.362763157888894,1.08124996423442,-0.224451023832774,0.182245928105777,0.33948396053001,-1.12207511166301,-0.016819823609436,1.48668878578401,0.385539537270371,0.280151681112708,-0.425599828473719,0.333255413834035,0.540875139707759,0.239039519774294,-1.25611131088952],
            [1.34444290589424,1.5483169711193,0.446663651432772,0.132943928767045,-1.00884158289578,-2.01292628499018,-0.17033402365269,0.377695926054327,0.16968500237739,-0.213226308819471,0.11514371371951,-0.0471737159523973,-0.750659045677889,0.513216269118057,-0.909783373460948,0.270418992067971,-1.81388524641712,-0.941151661350584,0.0540256614721632,0.586982747707026],
            [-1.28902025178048,-0.100866715203005,0.781923841997064,-0.783881334205479,1.78034155004277,0.729131330437947,0.0244871424164402,-1.34898168265066,-0.233050860188813,1.2776266286105,1.55453634596782,0.238320029897184,-0.0973158256420816,-0.63458099063283,-0.570514447990053,0.701497421838699,-0.00809768456597371,-0.130035057188228,-0.247517665855018,0.850898130742448],
            [-0.170024115106572,0.970310830276297,1.58246764869647,-2.14267612316755,0.408645173966463,-0.332850107772567,-0.209072599679776,-0.457102174195902,0.652770187923953,0.113018921250764,1.61818095550368,0.32701046372813,0.216363723477345,0.228006687583181,0.107103001976371,2.62803292316426,0.145095612238662,0.0743568691538312,-0.0989133029929123,0.453305771128927],
            [0.918315399833931,-0.528375452125217,2.71183856542826,3.20608668056192,1.8274677439393,-0.464751841066081,0.411587653885074,-0.49593941731022,1.00443993269949,-1.16521291418763,2.22067450666671,-1.791687725341,0.685235539329373,-0.559499114261227,-0.0701701733447188,0.32277361007293,-0.922994997685586,-0.433558484763715,0.0828804163188255,-1.429481573955],
            [1.42807120472356,0.445589077435416,-1.77588239294292,-0.637895116631973,-0.312470964574108,0.222550164050333,-0.961579137547036,1.06475393191926,-1.09220398808381,-0.106309340018392,1.06941968133478,0.115003113693731,-0.360374312696874,0.721237053224178,0.702202117701567,0.313524919137955,-0.321451695950242,-0.0155093515488093,0.699571645981518,-0.127601504058267],
            [0.57923560851419,0.164402567424472,0.365299058380659,0.86139208389801,0.274844453257317,0.0802823175772747,-0.643437872531412,-0.094279830138535,0.438587273147578,-0.355931672148484,0.548799941615759,1.05913698362718,-0.80464361550232,0.244940042923213,0.656268060196403,0.519165353376509,0.171394776787986,0.756547673845321,-0.412166815327595,-0.0105306494998952]
        ])
        self.eigenvalues = np.array([-0.667884055373576,-1.36231223042177,-1.03999021867912,0.952026509525691,-0.219079073976083,2.0833361372498,1.64407370601827,-0.289839100614106,0.201321690555994,-0.330655610801836,-1.24685903132477,-2.72341886367067,-1.02037365270831,1.58162607974058,0.206629576368476,0.672211071811841,0.57921458221257,-0.254188819951592,1.03608420414651,0.189844918151762])
        self.eigenvalues_2 = np.array([0.0972791839117805,-0.758546583836721,-0.541529878140039,-0.860033492077158,0.54469313179727,0.296806841342568,-0.692158537432337,-0.351782642612304,-1.76425910723177,-0.570236208194312,0.459385181765886,0.071876226745755,3.30452503914131,-0.191492835132291,-0.274643093756364])

    def test_find_optimal_alpha_dim_2(self):
        expected_output = 4.877329687500723
        output = _find_optimal_alpha(
           self.alpha_range, self.data, self.u, self.w,
           self.alpha, self.penalty_matrix,
           (self.eigenvalues, self.eigenvectors), 'i, j, ikj -> k'
        )
        np.testing.assert_almost_equal(output, expected_output, decimal=4)

    def test_find_optimal_alpha_dim_3(self):
        expected_output = 15.264197392928095
        output = _find_optimal_alpha(
           self.alpha_range, self.data, self.u, self.v,
           self.alpha, self.eigenvectors,
           (self.eigenvalues_2, self.penalty_matrix), 'i, j, ijk -> k'
        )
        np.testing.assert_almost_equal(output, expected_output, decimal=4)


class TestComputeDenominator(unittest.TestCase):
    def test_compute_denominator(self):
        v = np.array([1.30486965422349,2.28664539270111,-1.38886070111234,-0.278788766817371,-0.133321336393658,0.635950398070074,-0.284252921416072,-2.65645542090478,-2.44046692857552,1.32011334573019,-0.306638594078475,-1.78130843398,-0.171917355759621,1.2146746991726,1.89519346126497,-0.4304691316062,-0.25726938276893,-1.76316308519478,0.460097354831271,-0.639994875960119])
        alpha_v = 0.5
        mat_v = np.diff(np.identity(len(v)))
        penal_mat = np.dot(mat_v, mat_v.T)

        output = _compute_denominator(v, alpha_v, penal_mat)
        expected = 66.8605477690864
        np.testing.assert_almost_equal(output, expected)


class TestUpdateVector(unittest.TestCase):
    def setUp(self):
        a1 = np.linspace(0, 1, 10)
        a2 = np.linspace(-1, 1, 20)
        a3 = np.linspace(-np.pi, np.pi, 15)
        self.data = np.outer(np.outer(a1, a2), a3).reshape((10, 20, 15))

        S1 = self.data.shape[1]
        S2 = self.data.shape[2]
        self.alpha_range = {'v': (1e-4, 1e4), 'w': (1e-4, 1e4)}
        self.alphas = {'v': 0.5, 'w': 0.5}

        u = np.array([1.37095844714667,-0.564698171396089,0.363128411337339,0.63286260496104,0.404268323140999,-0.106124516091484,1.51152199743894,-0.0946590384130976,2.01842371387704,-0.062714099052421])
        v = np.array([1.30486965422349,2.28664539270111,-1.38886070111234,-0.278788766817371,-0.133321336393658,0.635950398070074,-0.284252921416072,-2.65645542090478,-2.44046692857552,1.32011334573019,-0.306638594078475,-1.78130843398,-0.171917355759621,1.2146746991726,1.89519346126497,-0.4304691316062,-0.25726938276893,-1.76316308519478,0.460097354831271,-0.639994875960119])
        w = np.array([0.455450123241219,0.704837337228819,1.03510352196992,-0.608926375407211,0.50495512329797,-1.71700867907334,-0.784459008379496,-0.850907594176518,-2.41420764994663,0.0361226068922556,0.205998600200254,-0.361057298548666,0.758163235699517,-0.726704827076575,-1.36828104441929])
        self.vectors = (u, v, w)

        mat_v = np.diff(np.identity(S1))
        mat_w = np.diff(np.identity(S2))
        self.penalty_matrices = {
            'v': np.dot(mat_v, mat_v.T),
            'w': np.dot(mat_w, mat_w.T)
        }

    def test_update_vector(self):
        v_cross = _compute_denominator(self.vectors[1], self.alphas['v'], self.penalty_matrices['v'])
        w_cross = _compute_denominator(self.vectors[2], self.alphas['w'], self.penalty_matrices['w'])
        results = _update_vector(self.data, self.vectors, 0, 0, v_cross * w_cross, 'i, j, kij -> k')
        expected = np.array([0., 0.00127016, 0.00254032, 0.00381048, 0.00508064, 0.00635081, 0.00762097, 0.00889113, 0.01016129, 0.01143145])

        np.testing.assert_array_almost_equal(results, expected)


class TestUpdateComponents(unittest.TestCase):
    def setUp(self):
        a1 = np.linspace(0, 1, 10)
        a2 = np.linspace(-1, 1, 20)
        a3 = np.linspace(-np.pi, np.pi, 15)
        self.data = np.outer(np.outer(a1, a2), a3).reshape((10, 20, 15))

        S1 = self.data.shape[1]
        S2 = self.data.shape[2]
        self.alpha_range = {'v': (1e-4, 1e4), 'w': (1e-4, 1e4)}
        self.alphas = {'v': 0.5, 'w': 0.5}

        u = np.array([1.37095844714667,-0.564698171396089,0.363128411337339,0.63286260496104,0.404268323140999,-0.106124516091484,1.51152199743894,-0.0946590384130976,2.01842371387704,-0.062714099052421])
        v = np.array([1.30486965422349,2.28664539270111,-1.38886070111234,-0.278788766817371,-0.133321336393658,0.635950398070074,-0.284252921416072,-2.65645542090478,-2.44046692857552,1.32011334573019,-0.306638594078475,-1.78130843398,-0.171917355759621,1.2146746991726,1.89519346126497,-0.4304691316062,-0.25726938276893,-1.76316308519478,0.460097354831271,-0.639994875960119])
        w = np.array([0.455450123241219,0.704837337228819,1.03510352196992,-0.608926375407211,0.50495512329797,-1.71700867907334,-0.784459008379496,-0.850907594176518,-2.41420764994663,0.0361226068922556,0.205998600200254,-0.361057298548666,0.758163235699517,-0.726704827076575,-1.36828104441929])
        self.vectors = (u, v, w)

        mat_v = np.diff(np.identity(S1))
        mat_w = np.diff(np.identity(S2))
        self.penalty_matrices = {
            'v': np.dot(mat_v, mat_v.T),
            'w': np.dot(mat_w, mat_w.T)
        }

        self.eigens = {
            'v': _eigh(self.penalty_matrices['v']),
            'w': _eigh(self.penalty_matrices['w'])
        }

    def test_update_components(self):
        results_vectors, results_alphas = _update_components(
            self.data,
            self.vectors,
            self.penalty_matrices,
            self.alphas,
            self.alpha_range,
            self.eigens
        )

        expected_u = np.array([0., 0.00127016, 0.00254032, 0.00381048, 0.00508064, 0.00635081, 0.00762097, 0.00889113, 0.01016129, 0.01143145])
        expected_v = np.array([ 29.0754956 ,  26.74521322,  23.79053357,  20.66854716, 17.50173105,  14.32290286,  11.14085607,   7.95794693, 4.77480702,   1.59160639,  -1.59160639,  -4.77480702, -7.95794693, -11.14085607, -14.32290286, -17.50173105, -20.66854716, -23.79053357, -26.74521322, -29.0754956])
        expected_w = np.array([ 7.80425621e+00,  6.94306630e+00,  5.85112128e+00,  4.69734572e+00, 3.52700313e+00,  2.35222294e+00,  1.17625939e+00, -3.52864950e-17, -1.17625939e+00, -2.35222294e+00, -3.52700313e+00, -4.69734572e+00, -5.85112128e+00, -6.94306630e+00, -7.80425621e+00])

        expected_alpha_v = 4.33996708058391
        expected_alpha_w = 3.366682373114449

        np.testing.assert_array_almost_equal(results_vectors[0], expected_u)
        np.testing.assert_array_almost_equal(results_vectors[1], expected_v)
        np.testing.assert_array_almost_equal(results_vectors[2], expected_w)
        np.testing.assert_almost_equal(results_alphas['v'], expected_alpha_v)
        np.testing.assert_almost_equal(results_alphas['w'], expected_alpha_w)


class FCPTPATest(unittest.TestCase):
    def setUp(self):
        kl = KarhunenLoeve(
            basis_name='bsplines',
            n_functions=5,
            dimension='2D',
            argvals=np.linspace(0, 1, 10),
            random_state=42
        )
        kl.new(n_obs=50)
        self.data = kl.data

        n_points = self.data.n_points
        mat_v = np.diff(np.identity(n_points[0]))
        mat_w = np.diff(np.identity(n_points[1]))
        self.penalty_matrices={
            'v': np.dot(mat_v, mat_v.T),
            'w': np.dot(mat_w, mat_w.T)
        }
        self.alpha_range = {'v': (1e-2, 1e2), 'w': (1e-2, 1e2)}

        self.fcp = FCPTPA()
        self.fcp.fit(
            self.data,
            penalty_matrices=self.penalty_matrices,
            alpha_range=self.alpha_range,
            tolerance=1e-4,
            max_iteration=15,
            adapt_tolerance=True,
            verbose=True
        )


    def test_init(self):
        # Test default initialization
        fcp = FCPTPA()
        self.assertEqual(fcp.n_components, 5)
        self.assertFalse(fcp.normalize)

        # Test custom initialization
        fcp = FCPTPA(n_components=3, normalize=True)
        self.assertEqual(fcp.n_components, 3)
        self.assertTrue(fcp.normalize)

    def test_n_components(self):
        fcp = FCPTPA()
        fcp.n_components = 4
        self.assertEqual(fcp.n_components, 4)

    def test_normalize(self):
        fcp = FCPTPA()
        fcp.normalize = True
        self.assertTrue(fcp.normalize)

    def test_eigenvalues(self):
        self.assertIsInstance(self.fcp.eigenvalues, np.ndarray)

        with self.assertRaises(AttributeError):
            self.fcp.eigenvalues = 1  # Can't set attributes

    def test_eigenfunctions(self):
        self.assertIsInstance(self.fcp.eigenfunctions, DenseFunctionalData)

        with self.assertRaises(AttributeError):
            self.fcp.eigenfunctions = 1  # Can't set attributes


class TestFit(unittest.TestCase):
    def setUp(self):
        kl = KarhunenLoeve(
            basis_name='bsplines',
            n_functions=5,
            dimension='2D',
            argvals=np.linspace(0, 1, 10),
            random_state=42
        )
        kl.new(n_obs=50)
        self.data = kl.data

        n_points = self.data.n_points
        mat_v = np.diff(np.identity(n_points[0]))
        mat_w = np.diff(np.identity(n_points[1]))
        self.penalty_matrices={
            'v': np.dot(mat_v, mat_v.T),
            'w': np.dot(mat_w, mat_w.T)
        }
        self.alpha_range = {'v': (1e-2, 1e2), 'w': (1e-2, 1e2)}

    def test_fit(self):
        fcptpa = FCPTPA(n_components=5)
        fcptpa.fit(
            self.data,
            penalty_matrices=self.penalty_matrices,
            alpha_range=self.alpha_range,
            tolerance=1e-4,
            max_iteration=15,
            adapt_tolerance=True,
            verbose=True
        )

        np.testing.assert_equal(fcptpa._scores.shape, (50, 5))
        np.testing.assert_equal(fcptpa.eigenvalues.shape, (5,))
        self.assertIsInstance(fcptpa.eigenfunctions, DenseFunctionalData)

    def test_fit_warnings_convergence(self):
        fcptpa = FCPTPA(n_components=5)
        with self.assertWarns(UserWarning):
            fcptpa.fit(
                self.data,
                penalty_matrices=self.penalty_matrices,
                alpha_range={'v': (1e-2, 1e2), 'w': (1e-2, 1e2)},
                tolerance=1e-10,
                max_iteration=1,
                adapt_tolerance=True
            )


class TestFitNorm(unittest.TestCase):
    def setUp(self):
        kl = KarhunenLoeve(
            basis_name='bsplines',
            n_functions=5,
            dimension='2D',
            argvals=np.linspace(0, 1, 10),
            random_state=42
        )
        kl.new(n_obs=50)
        self.data = kl.data

        n_points = self.data.n_points
        mat_v = np.diff(np.identity(n_points[0]))
        mat_w = np.diff(np.identity(n_points[1]))
        self.penalty_matrices={
            'v': np.dot(mat_v, mat_v.T),
            'w': np.dot(mat_w, mat_w.T)
        }
        self.alpha_range = {'v': (1e-2, 1e2), 'w': (1e-2, 1e2)}

    def test_fit_norm(self):
        fcptpa = FCPTPA(n_components=5, normalize=True)
        fcptpa.fit(
            self.data,
            penalty_matrices=self.penalty_matrices,
            alpha_range=self.alpha_range,
            tolerance=1e-4,
            max_iteration=15,
            adapt_tolerance=True,
            verbose=True
        )

        np.testing.assert_equal(fcptpa._scores.shape, (50, 5))
        np.testing.assert_equal(fcptpa.eigenvalues.shape, (5,))
        self.assertIsInstance(fcptpa.eigenfunctions, DenseFunctionalData)


class TestTransform(unittest.TestCase):
    def setUp(self):
        kl = KarhunenLoeve(
            basis_name='bsplines',
            n_functions=5,
            dimension='2D',
            argvals=np.linspace(0, 1, 10),
            random_state=42
        )
        kl.new(n_obs=50)
        self.data = kl.data

        n_points = self.data.n_points
        mat_v = np.diff(np.identity(n_points[0]))
        mat_w = np.diff(np.identity(n_points[1]))

        self.fcptpa = FCPTPA(n_components=5)
        self.fcptpa.fit(
            self.data,
            penalty_matrices={'v': np.dot(mat_v, mat_v.T), 'w': np.dot(mat_w, mat_w.T)},
            alpha_range={'v': (1e-2, 1e2), 'w': (1e-2, 1e2)},
            tolerance=1e-4,
            max_iteration=15,
            adapt_tolerance=True
        )

    def test_transform_numint(self):
        # We only test the shape of the output because the optimization step
        # can lead to different solution
        scores = self.fcptpa.transform(self.data, method='NumInt')
        expected_shape = (50, 5)
        np.testing.assert_array_equal(scores.shape, expected_shape)

    def test_transform_numint_norm(self):
        # We only test the shape of the output because the optimization step
        # can lead to different solution
        n_points = self.data.n_points
        mat_v = np.diff(np.identity(n_points[0]))
        mat_w = np.diff(np.identity(n_points[1]))

        fcptpa = FCPTPA(n_components=5, normalize=True)
        fcptpa.fit(
            self.data,
            penalty_matrices={'v': np.dot(mat_v, mat_v.T), 'w': np.dot(mat_w, mat_w.T)},
            alpha_range={'v': (1e-2, 1e2), 'w': (1e-2, 1e2)},
            tolerance=1e-4,
            max_iteration=15,
            adapt_tolerance=True
        )
    
        scores = fcptpa.transform(self.data, method='NumInt')
        expected_shape = (50, 5)
        np.testing.assert_array_equal(scores.shape, expected_shape)

    def test_transform_fcptpa(self):
        # We only test the shape of the output because the optimization step
        # can lead to different solution
        scores = self.fcptpa.transform(self.data, method='FCPTPA')
        expected_shape = (50, 5)
        np.testing.assert_array_equal(scores.shape, expected_shape)

    def test_transform_error(self):
        with self.assertRaises(ValueError):
            self.fcptpa.transform(self.data, method='error')


class TestInverseTransform(unittest.TestCase):
    def setUp(self):
        kl = KarhunenLoeve(
            basis_name='bsplines',
            n_functions=5,
            dimension='2D',
            argvals=np.linspace(0, 1, 10),
            random_state=42
        )
        kl.new(n_obs=50)
        self.data = kl.data

        n_points = self.data.n_points
        mat_v = np.diff(np.identity(n_points[0]))
        mat_w = np.diff(np.identity(n_points[1]))

        self.fcptpa = FCPTPA(n_components=5)
        self.fcptpa.fit(
            self.data,
            penalty_matrices={'v': np.dot(mat_v, mat_v.T), 'w': np.dot(mat_w, mat_w.T)},
            alpha_range={'v': (1e-2, 1e2), 'w': (1e-2, 1e2)},
            tolerance=1e-4,
            max_iteration=15,
            adapt_tolerance=True
        )
        self.scores = self.fcptpa.transform(self.data)

    def test_inverse_transform(self):
        data_f = self.fcptpa.inverse_transform(self.scores)
        self.assertIsInstance(data_f, DenseFunctionalData)
        np.testing.assert_equal(self.data.argvals, data_f.argvals)
        np.testing.assert_equal(self.data.values.shape, data_f.values.shape)
