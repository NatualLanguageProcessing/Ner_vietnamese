jQuery(document).ready(function() {
	'use strict';

	Morris.Area({
		element: "area_line_chart",
		behaveLikeLine: true,
		data: [
		       {w: '2011 Q1', x: 2, y: 0, z: 0},
		       {w: '2011 Q2', x: 50, y: 15, z: 5},
		       {w: '2011 Q3', x: 15, y: 50, z: 23},
		       {w: '2011 Q4', x: 45, y: 12, z: 7},
		       {w: '2011 Q5', x: 20, y: 32, z: 55},
		       {w: '2011 Q6', x: 39, y: 67, z: 20},
		       {w: '2011 Q7', x: 20, y: 9, z: 5}
		       ],
		       xkey: 'w',
		       ykeys: ['x', 'y', 'z'],
		       labels: ['X', 'Y', 'Z'],
		       pointSize: 2,
		       hideHover: 'auto',
		       lineColors: ['rgb(97, 97, 97)', 'rgb(0, 206, 209)', 'rgb(255, 117, 142)']
	});

	Morris.Line({
		element: "line_chart",
		data: [
		       {"period": "2012-10-01", "licensed": 3407, "sorned": 660},
		       {"period": "2012-09-30", "licensed": 3351, "sorned": 629},
		       {"period": "2012-09-29", "licensed": 3269, "sorned": 618},
		       {"period": "2012-09-20", "licensed": 3246, "sorned": 661},
		       {"period": "2012-09-19", "licensed": 3257, "sorned": 667},
		       {"period": "2012-09-18", "licensed": 3248, "sorned": 627},
		       {"period": "2012-09-17", "licensed": 3171, "sorned": 660},
		       {"period": "2012-09-16", "licensed": 3171, "sorned": 676},
		       {"period": "2012-09-15", "licensed": 3201, "sorned": 656},
		       {"period": "2012-09-10", "licensed": 3215, "sorned": 622}
		       ],
		       xkey: 'period',
		       ykeys: ['licensed', 'sorned'],
		       labels: ['Licensed', 'SORN']
	});

	Morris.Bar({
		element: "bar_chart",
		data: [
		       {x: '2011 Q1', y: 3, z: 2, a: 3},
		       {x: '2011 Q2', y: 2, z: null, a: 1},
		       {x: '2011 Q3', y: 0, z: 2, a: 4},
		       {x: '2011 Q4', y: 2, z: 4, a: 3}
		       ],
		       xkey: 'x',
		       ykeys: ['y', 'z', 'a'],
		       labels: ['Y', 'Z', 'A'],
		       barColors: ['rgb(233, 30, 99)', 'rgb(0, 188, 212)', 'rgb(0, 150, 136)']
	}).on('click', function(i, row){
		console.log(i, row);
	});
	Morris.Area({
		element: "area_chart",
		data: [{
			period: '2010 Q1',
			iphone: 2666,
			ipad: null,
			itouch: 2647
		}, {
			period: '2010 Q2',
			iphone: 2778,
			ipad: 2294,
			itouch: 2441
		}, {
			period: '2010 Q3',
			iphone: 4912,
			ipad: 1969,
			itouch: 2501
		}, {
			period: '2010 Q4',
			iphone: 3767,
			ipad: 3597,
			itouch: 5689
		}, {
			period: '2011 Q1',
			iphone: 6810,
			ipad: 1914,
			itouch: 2293
		}, {
			period: '2011 Q2',
			iphone: 5670,
			ipad: 4293,
			itouch: 1881
		}, {
			period: '2011 Q3',
			iphone: 4820,
			ipad: 3795,
			itouch: 1588
		}, {
			period: '2011 Q4',
			iphone: 15073,
			ipad: 5967,
			itouch: 5175
		}, {
			period: '2012 Q1',
			iphone: 10687,
			ipad: 4460,
			itouch: 2028
		}, {
			period: '2012 Q2',
			iphone: 8432,
			ipad: 5713,
			itouch: 1791
		}],
		xkey: 'period',
		ykeys: ['iphone', 'ipad', 'itouch'],
		labels: ['iPhone', 'iPad', 'iPod Touch'],
		pointSize: 2,
		hideHover: 'auto',
		lineColors: ['rgb(233, 30, 99)', 'rgb(0, 188, 212)', 'rgb(0, 150, 136)']
	});
	Morris.Donut({
		element: "donut_chart",
		data: [{
			label: 'Jam',
			value: 25
		}, {
			label: 'Frosted',
			value: 40
		}, {
			label: 'Custard',
			value: 25
		}, {
			label: 'Sugar',
			value: 10
		}],
		colors: ['rgb(233, 30, 99)', 'rgb(0, 188, 212)', 'rgb(255, 152, 0)', 'rgb(0, 150, 136)'],
		formatter: function (y) {
			return y + '%'
		}
	});
});


