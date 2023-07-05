/*
  Theme Name: Rex-Slider jQuery Slider in HTML5 Plugin
  Author: ThemexHunter
  Description: jQuery Slider in HTML5 Plugin
  Version: 1.0
*/


(function ($) {
    "use strict";

	
/********* MOBILE NAV *********/	
$(document).ready(function(){
	$('#nav').slicknav();
	$(".slicknav_menu").appendTo(".mainmenu_area_2");	
});
	
	
$('.preloader').delay(1000).fadeOut(500, function() {});
	
	
/* POPUP VIDEO */  
if ($(".video-btn").length) {
	$(".video-btn").on("click", function(){
		$.fancybox({
			href: this.href,
			type: $(this).data("type"),
			'title'         : this.title,
			helpers     : {  
				title : { type : 'inside' },
				media : {}
			},

			beforeShow : function(){
				$(".fancybox-wrap").addClass("gallery-fancybox");
			}
		});
		return false
	});    
}	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

})(jQuery);