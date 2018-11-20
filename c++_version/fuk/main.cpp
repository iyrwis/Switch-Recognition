//#include <dlib/svm_threaded.h>
//#include <dlib/string.h>
//#include <dlib/gui_widgets.h>
//#include <dlib/image_processing.h>
//#include <dlib/data_io.h>
//#include <dlib/cmd_line_parser.h>


//#include <iostream>
//#include <fstream>


//using namespace std;
//using namespace dlib;

//const int threads = 4;
//const double C = 1.0;
//const double eps = 0.01;

//int t=0;

//int main()
//{
//    dlib::array<array2d<unsigned char> > images_train, images_test;
//    std::vector<std::vector<rectangle> > object_locations_train, object_locations_test;

//    load_image_dataset(images_train, object_locations_train,"./switch_loc_train/data.xml");
//    load_image_dataset(images_test, object_locations_test,"./switch_loc_test/data2.xml");
    //upsample_image_dataset<pyramid_down<2> >(images_train, object_locations_train);
    //upsample_image_dataset<pyramid_down<2> >(images_test, object_locations_test);

//    cout<<"The number of train images: "<<images_train.size()<<endl;
//    cout<<"The number of test images: "<<images_test.size()<<endl;

//    typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;
//    image_scanner_type scanner;
//    unsigned long width=80, height=80;
//    scanner.set_detection_window_size(width, height);

//    structural_object_detection_trainer<image_scanner_type> trainer(scanner);
//    trainer.set_num_threads(threads);
//    trainer.set_c(C);
//    trainer.be_verbose();
//    trainer.set_epsilon(eps);
//    object_detector<image_scanner_type> detector;

//    if(!t)
//    {
//        detector = trainer.train(images_train, object_locations_train);
//        cout<<"Saving trained detector to object_detector.svm"<<endl;
//        serialize("./object_detector.svm")<<detector;
//        cout << "training results: " << test_object_detection_function(detector, images_train, object_locations_train) << endl;
//        cout << "testing results:  " << test_object_detection_function(detector, images_test, object_locations_test) << endl;
//    }

    //for(unsigned long i=0; i<images_.size(); i++)
    //{
    //    const std::vector<rectangle> rects = detector(images[i]);
    //    cout<<"Number of detections: "<<rects.size()<<endl;
    //}

//    image_window hogwin(draw_fhog(detector), "Learned fHOG detector");
//    image_window win;
//    for (unsigned long i = 0; i < images_test.size(); ++i)
//    {
//        std::vector<rectangle> dets = detector(images_test[i]);
//        win.clear_overlay();
//        win.set_image(images_test[i]);
//        win.add_overlay(dets, rgb_pixel(255,0,0));
//        cout << "Hit enter to process the next image..." << endl;
//        cin.get();
//     }
//    return 0;
//}





